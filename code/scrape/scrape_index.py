"""Scrape - Step 1: build an index of every car listing currently on ikman.lk.

Walks the paginated search results and records one line per unique ad. This gives
us the slug needed to fetch each ad's detail page in step 2, plus the summary
fields the search page already provides.

Deduplication matters here: ikman injects promoted listings into multiple pages,
so the same ad id genuinely appears on page 1 and page 400. Counting rows without
deduplicating would inflate the dataset and duplicate records across the
train/test split - which would leak test data into training and quietly inflate
every score.

Run:  python code/scrape/scrape_index.py [--pages N] [--delay SECONDS]
Output: data/raw/index.jsonl  (append-only, resumable)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import sys
from pathlib import Path

import requests

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ikman

OUT = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'index.jsonl'


def summarise(ad: dict) -> dict:
    """Keep the fields we may want; drop seller identity and image payloads."""
    return {
        'id': ad['id'],
        'slug': ad['slug'],
        'title': ad.get('title'),
        'price_raw': ad.get('price'),
        'location': ad.get('location'),
        'details_raw': ad.get('details'),
        'category': (ad.get('category') or {}).get('name'),
        'is_verified': ad.get('isVerified'),
        'is_member': ad.get('isMember'),
        'is_promoted': ad.get('isFeaturedAd') or ad.get('isTopAd'),
        'seen_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--pages', type=int, default=0,
                   help='pages to walk; 0 means derive from the reported total')
    p.add_argument('--delay', type=float, default=1.0, help='seconds between requests')
    p.add_argument('--category', default='cars')
    args = p.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = ikman.read_jsonl(OUT)
    seen: set[str] = {r['id'] for r in existing}
    print(f'index already holds {len(seen)} unique ads')

    session = requests.Session()

    # First page tells us how many there are in total.
    data = ikman.get(session, ikman.search_page_url(1, args.category), delay=args.delay)
    serp = data['serp']['ads']['data']
    total = serp['paginationData']['total']
    per_page = len(serp['ads'])
    pages = args.pages or (total // per_page + 2)
    print(f'site reports {total} ads, {per_page} per page -> walking {pages} pages')

    added = 0
    with open(OUT, 'a', encoding='utf-8') as f:
        for page in range(1, pages + 1):
            if page == 1:
                ads = serp['ads']
            else:
                try:
                    d = ikman.get(session, ikman.search_page_url(page, args.category),
                                  delay=args.delay)
                    ads = d['serp']['ads']['data']['ads']
                except ikman.FetchError as e:
                    print(f'  page {page}: {e}')
                    continue

            new = 0
            for ad in ads:
                if ad['id'] in seen:
                    continue
                seen.add(ad['id'])
                f.write(json.dumps(summarise(ad), ensure_ascii=False) + '\n')
                new += 1
            added += new
            f.flush()

            if page % 10 == 0 or page <= 3:
                print(f'  page {page:>4}/{pages}  +{new:>2} new  '
                      f'unique so far {len(seen)}')

            if not ads:
                print(f'  page {page}: empty, stopping')
                break

    print(f'\ndone: {added} new ads added, {len(seen)} unique in total')
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
