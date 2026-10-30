"""Scrape - Step 2: fetch each indexed ad's detail page for its full property set.

The search page only carries price, location, mileage, body and condition. The
detail page adds brand, model, edition, year, transmission, fuel type and engine
capacity - the features that make the model worth explaining.

Values are stored exactly as the site reports them (`"35,000 km"`, `"1,000 cc"`,
`"Rs 8,700,000"`). Parsing happens in a separate cleaning step so that this
script stays a faithful record of what was collected, and a parsing bug never
means re-scraping.

Run:  python code/scrape/scrape_details.py [--limit N] [--delay SECONDS] [--workers N]
Output: data/raw/details.jsonl  (append-only, resumable)
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from datetime import datetime, timezone
import sys
from pathlib import Path
from queue import Queue

import requests

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ikman

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'data' / 'raw' / 'index.jsonl'
OUT = ROOT / 'data' / 'raw' / 'details.jsonl'

# Fields kept from the ad record. Deliberately excludes `account`, `memberLogo`,
# phone numbers and image ids: none of it helps predict price, and seller
# identity is exactly what should not be sitting in a student dataset.
KEEP = (
    'id', 'slug', 'title', 'postedDate', 'adDate',
    'isVerified', 'isMember', 'isAuthDealer', 'isPromoted', 'membershipLevel',
)


def flatten(ad: dict) -> dict:
    row: dict = {k: ad.get(k) for k in KEEP}

    # properties is a list of {key, label, value}; spread it into columns.
    for p in ad.get('properties') or []:
        if p.get('key'):
            row[f'prop_{p["key"]}'] = p.get('value')

    money = ad.get('money') or {}
    row['price_raw'] = money.get('amount')

    # `location` nests the full boundary polygon of every town and its parent
    # district - tens of coordinate pairs per ad. Keep only the three fields that
    # carry information, or the raw file balloons by an order of magnitude for
    # data no model will ever use.
    loc = ad.get('location') or {}
    row['town'] = loc.get('name')
    row['district'] = (loc.get('parent') or {}).get('name') or (ad.get('area') or {}).get('name')
    row['geo_region'] = loc.get('geo_region')

    row['views'] = (ad.get('statistics') or {}).get('views')
    row['fetched_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return row


def worker(q: Queue, session_local: threading.local, out_lock: threading.Lock,
           fh, delay: float, counters: dict) -> None:
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            return

        session = getattr(session_local, 'session', None)
        if session is None:
            session = requests.Session()
            session_local.session = session

        try:
            data = ikman.get(session, ikman.ad_url(item['slug']), delay=delay)
            if not data:
                counters['gone'] += 1
            else:
                ad = data['adDetail']['data']['ad']
                row = flatten(ad)
                with out_lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                    fh.flush()
                    counters['ok'] += 1
        except (ikman.FetchError, KeyError, TypeError) as e:
            counters['failed'] += 1
            if counters['failed'] <= 5:
                print(f'  ! {item["slug"][:60]}: {type(e).__name__} {e}')

        done = counters['ok'] + counters['failed'] + counters['gone']
        if done % 100 == 0:
            print(f'  {done:>5} processed  ok={counters["ok"]} '
                  f'gone={counters["gone"]} failed={counters["failed"]}')
        q.task_done()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0, help='max ads to fetch; 0 = all')
    p.add_argument('--delay', type=float, default=1.0,
                   help='seconds each worker waits before a request')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    index = ikman.read_jsonl(INDEX)
    if not index:
        raise SystemExit(f'no index found at {INDEX} - run scrape_index.py first')

    done_ids = {r['id'] for r in ikman.read_jsonl(OUT)}
    todo = [r for r in index if r['id'] not in done_ids]

    # Shuffle with a fixed seed so that stopping early leaves a random sample of
    # the whole index rather than a block ordered by recency or by price.
    # Recency ordering would make an early stop a biased sample.
    random.Random(args.seed).shuffle(todo)

    if args.limit:
        todo = todo[:args.limit]

    print(f'index: {len(index)} ads | already fetched: {len(done_ids)} | '
          f'to fetch now: {len(todo)}')
    if not todo:
        return

    est = len(todo) * args.delay / max(args.workers, 1) / 60
    print(f'estimated wall time: ~{est:.0f} min at delay={args.delay}s '
          f'x {args.workers} workers')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    q: Queue = Queue()
    counters = {'ok': 0, 'failed': 0, 'gone': 0}
    out_lock = threading.Lock()
    session_local = threading.local()

    with open(OUT, 'a', encoding='utf-8') as fh:
        threads = [
            threading.Thread(target=worker,
                             args=(q, session_local, out_lock, fh, args.delay, counters),
                             daemon=True)
            for _ in range(args.workers)
        ]
        for t in threads:
            t.start()
        for item in todo:
            q.put(item)
        for _ in threads:
            q.put(None)
        q.join()

    print(f'\ndone: ok={counters["ok"]} gone={counters["gone"]} '
          f'failed={counters["failed"]}')
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
