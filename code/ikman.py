"""Shared helpers for collecting vehicle listings from ikman.lk.

ikman.lk serves each page as server-rendered HTML with the whole page state
embedded in a single `window.initialData = {...}` assignment. Reading that JSON
is far more stable than parsing the HTML, because the JSON is the same structure
the site's own front-end consumes, whereas CSS classes change whenever the site
is restyled.

Collection policy
-----------------
ikman.lk/robots.txt allows `/` for all user agents (it names several AI crawlers
explicitly) and disallows only URLs carrying `utm_source`, `c=`, `account=` or
`filters=` query parameters. The URLs used here carry none of those. Requests are
rate limited and retried with backoff so the site is never hit hard, and no
seller contact details are collected.
"""

from __future__ import annotations

import json
import random
import time

import requests

BASE = 'https://ikman.lk'

# A desktop browser UA. ikman serves the same server-rendered payload either way;
# this just avoids being bucketed as an unknown client.
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}

MARKER = 'window.initialData'


class FetchError(RuntimeError):
    pass


def extract_initial_data(html: str) -> dict:
    """Pull the `window.initialData` JSON object out of a page.

    Scans forward from the marker and matches braces while tracking string
    literals and escapes, so a `{` or `}` inside a listing description cannot
    end the object early. A plain regex would break on exactly that.
    """
    i = html.find(MARKER)
    if i == -1:
        raise FetchError(f'{MARKER} not found - page shape changed or request was blocked')

    start = html.find('{', i)
    depth = 0
    in_string = False
    escaped = False

    for j in range(start, len(html)):
        c = html[j]
        if in_string:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return json.loads(html[start:j + 1])

    raise FetchError('unbalanced braces while reading initialData')


def get(session: requests.Session, url: str, *, delay: float = 1.0,
        attempts: int = 4) -> dict:
    """Fetch a page and return its initialData, politely and with retries.

    `delay` is applied *before* each request, with a little jitter, so a long run
    does not arrive as a perfectly regular machine-gun pattern. Retries back off
    exponentially; 404 is returned to the caller as None rather than retried,
    since an expired listing will never succeed.
    """
    last_error: Exception | None = None

    for attempt in range(attempts):
        time.sleep(delay * (1 + random.uniform(-0.25, 0.25)) + attempt * delay * 2)
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            last_error = e
            continue

        if r.status_code == 404:
            return {}
        if r.status_code == 200:
            # The pages are UTF-8, but ikman does not put a charset in the
            # Content-Type header, and requests then falls back to ISO-8859-1 for
            # text/* as RFC 2616 tells it to. Reading `.text` without this line
            # decodes UTF-8 bytes as latin-1 and silently mangles every non-ASCII
            # character: 'Coupe/Sports' arrives as 'CoupA(c)/Sports', '4x4' as
            # '4A-4', the degree sign in '360deg Camera' as 'A°'. It corrupted 83
            # rows of the first scrape before it was noticed in a figure.
            r.encoding = 'utf-8'
            try:
                return extract_initial_data(r.text)
            except FetchError as e:
                last_error = e
                continue
        last_error = FetchError(f'HTTP {r.status_code} for {url}')

    raise FetchError(f'giving up on {url}: {last_error}')


def search_page_url(page: int, category: str = 'cars') -> str:
    return f'{BASE}/en/ads/sri-lanka/{category}?page={page}'


def ad_url(slug: str) -> str:
    return f'{BASE}/en/ad/{slug}'


def read_jsonl(path) -> list[dict]:
    """Read a JSONL file, tolerating a truncated final line from an interrupted run."""
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # partial write at the tail of an interrupted run
    except FileNotFoundError:
        pass
    return rows
