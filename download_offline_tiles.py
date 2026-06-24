"""Pre-download Google Satellite tiles for offline use, same idea as the
QGroundControl / Mission Planner offline map cache.

Run this ONCE while you still have internet, before flying offline. Tiles are
written to tile_cache/<z>/<x>/<y>.png which server.py's /tiles/<z>/<x>/<y>.png
route serves first (offline), falling back to live fetch only when online.

Example (default center is the project's GPS station, 900m radius):
    python download_offline_tiles.py
    python download_offline_tiles.py --lat 10.850970 --lon 106.771961 --radius 900 --zoom-min 14 --zoom-max 19

Notes / why the old run hung:
  * --zoom-max 21 over a 900m radius is ~12,000 tiles (z20=2401, z21=9409).
    Default max is now 19 (~880 tiles total). Add --zoom-max 20/21 only if you
    really need to zoom in that far, and expect it to take a while.
  * If you are OFFLINE the script now fails fast with a clear message instead
    of hanging on DNS for every tile.
"""
import argparse
import math
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter

TILE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tile_cache')
TILE_HOST = 'mt0.google.com'


def deg2num(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def bbox_from_radius(lat, lon, radius_m):
    lat_delta = radius_m / 111320.0
    lon_delta = radius_m / (111320.0 * math.cos(math.radians(lat)))
    return lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta  # south, west, north, east


def check_online(host=TILE_HOST, port=443, timeout=5):
    """Quick connectivity preflight so we fail fast (not hang on DNS) offline."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(host, port)  # DNS first (this is what hung before)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
    finally:
        socket.setdefaulttimeout(None)


def plan_tiles(lat, lon, radius_m, zoom_min, zoom_max, out_dir):
    """Build the full (z, x, y, path) work list and report per-zoom counts."""
    south, west, north, east = bbox_from_radius(lat, lon, radius_m)
    jobs = []
    for z in range(zoom_min, zoom_max + 1):
        x_min, y_max = deg2num(south, west, z)
        x_max, y_min = deg2num(north, east, z)
        x_lo, x_hi = min(x_min, x_max), max(x_min, x_max)
        y_lo, y_hi = min(y_min, y_max), max(y_min, y_max)
        count = (x_hi - x_lo + 1) * (y_hi - y_lo + 1)
        print(f'Zoom {z:2d}: {count} tiles (x {x_lo}-{x_hi}, y {y_lo}-{y_hi})')
        for x in range(x_lo, x_hi + 1):
            for y in range(y_lo, y_hi + 1):
                tile_dir = os.path.join(out_dir, str(z), str(x))
                tile_path = os.path.join(tile_dir, f'{y}.png')
                jobs.append((z, x, y, tile_dir, tile_path))
    return jobs


def make_session(pool_size):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def fetch_tile(session, z, x, y, tile_dir, tile_path, timeout=10, attempts=3):
    """Download one tile. Returns 'new', 'skip', or 'fail'."""
    if os.path.exists(tile_path) and os.path.getsize(tile_path) > 0:
        return 'skip'
    subdomain = ['mt0', 'mt1', 'mt2', 'mt3'][(x + y) % 4]
    url = f'https://{subdomain}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    for attempt in range(attempts):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.headers.get('content-type', '').startswith('image'):
                os.makedirs(tile_dir, exist_ok=True)
                # Write atomically so a Ctrl+C mid-write never leaves a half tile.
                tmp = tile_path + '.part'
                with open(tmp, 'wb') as f:
                    f.write(r.content)
                os.replace(tmp, tile_path)
                return 'new'
            # Non-image / error response: brief backoff then retry.
            time.sleep(0.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(0.5 * (attempt + 1))
    return 'fail'


def download_tiles(lat, lon, radius_m, zoom_min, zoom_max, out_dir, workers=8):
    jobs = plan_tiles(lat, lon, radius_m, zoom_min, zoom_max, out_dir)
    total = len(jobs)
    print(f'\nTotal {total} tiles to check across zoom {zoom_min}-{zoom_max}.')

    if total > 5000:
        print(f'WARNING: {total} tiles is a lot. Consider a lower --zoom-max '
              f'or smaller --radius. Continuing in 3s (Ctrl+C to abort)...')
        time.sleep(3)

    if not check_online():
        print('\nERROR: No internet connection (cannot reach Google tile server).')
        print('Connect to the network and run this again while ONLINE to build '
              'the offline cache.')
        return 1

    session = make_session(workers)
    counts = {'new': 0, 'skip': 0, 'fail': 0}
    counts_lock = threading.Lock()
    done = 0
    failed_tiles = []

    print(f'Downloading with {workers} workers...\n')
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_tile, session, z, x, y, td, tp): (z, x, y)
                for (z, x, y, td, tp) in jobs
            }
            for fut in as_completed(futures):
                result = fut.result()
                with counts_lock:
                    counts[result] += 1
                    done += 1
                    cur_done, cur_new, cur_skip, cur_fail = done, counts['new'], counts['skip'], counts['fail']
                if result == 'fail':
                    failed_tiles.append(futures[fut])
                if cur_done % 25 == 0 or cur_done == total:
                    pct = 100.0 * cur_done / total if total else 100.0
                    sys.stdout.write(
                        f'\r  {cur_done}/{total} ({pct:5.1f}%)  '
                        f'new={cur_new} cached={cur_skip} fail={cur_fail}   '
                    )
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print('\nInterrupted by user. Partial cache kept; re-run to resume '
              '(already-downloaded tiles are skipped).')
        return 130

    elapsed = time.time() - start
    print(f'\n\nDone in {elapsed:.0f}s. Downloaded {counts["new"]} new, '
          f'{counts["skip"]} already cached, {counts["fail"]} failed. Saved to {out_dir}')
    if failed_tiles:
        print(f'{len(failed_tiles)} tiles failed (e.g. {failed_tiles[:5]}). '
              f'Re-run while online to retry just those.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-download offline map tiles around a GPS point.')
    parser.add_argument('--lat', type=float, default=10.850970)
    parser.add_argument('--lon', type=float, default=106.771961)
    parser.add_argument('--radius', type=float, default=900, help='radius in meters (default 900m, under 1km)')
    parser.add_argument('--zoom-min', type=int, default=2)
    parser.add_argument('--zoom-max', type=int, default=21,
                        help='max zoom (default 19 ~880 tiles; 20/21 add thousands more)')
    parser.add_argument('--workers', type=int, default=8, help='parallel download threads')
    parser.add_argument('--out', type=str, default=TILE_CACHE_DIR)
    args = parser.parse_args()

    if args.zoom_min > args.zoom_max:
        parser.error('--zoom-min must be <= --zoom-max')

    rc = download_tiles(args.lat, args.lon, args.radius, args.zoom_min, args.zoom_max,
                        args.out, workers=args.workers)
    sys.exit(rc)
