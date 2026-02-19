import argparse
import os
import random
import sys
import urllib.request
import urllib.parse

BASE = 'https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg'

BASE_POINTS = [
  (202.467917, 47.198333),
  (187.705833, 12.391111),
  (210.804167, 54.348056),
  (224.594100, -1.090000),
]

def fetch(url, outpath):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    with open(outpath,'wb') as f:
        f.write(data)

def maybe_good_jpeg(path, min_bytes=8000):
    try:
        return os.path.getsize(path) >= min_bytes
    except Exception:
        return False

def download_set(outdir, n, seed=0):
    random.seed(seed)
    os.makedirs(outdir, exist_ok=True)
    scale = 0.25
    width = 128
    height = 128

    k = 0
    attempts = 0
    while k < n and attempts < n*20:
        attempts += 1
        bra, bdec = random.choice(BASE_POINTS)
        # small random offset in degrees
        ra  = bra  + random.uniform(-0.08, 0.08)
        dec = bdec + random.uniform(-0.08, 0.08)
        params = dict(ra=ra, dec=dec, scale=scale, width=width, height=height, opt='')
        url = BASE + '?' + urllib.parse.urlencode(params)
        fname = f'cutout_{k:04d}.jpg'
        outpath = os.path.join(outdir, fname)
        tmp = outpath + '.tmp'
        try:
            fetch(url, tmp)
            if maybe_good_jpeg(tmp):
                os.replace(tmp, outpath)
                k += 1
            else:
                try: os.remove(tmp)
                except Exception: pass
        except Exception:
            try: os.remove(tmp)
            except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_out', required=True)
    ap.add_argument('--test_out', required=True)
    ap.add_argument('--n_train', type=int, default=200)
    ap.add_argument('--n_test', type=int, default=60)
    args = ap.parse_args()

    print('Downloading raw SDSS cutouts...')
    download_set(args.train_out, args.n_train, seed=0)
    download_set(args.test_out, args.n_test, seed=1)
    print('Done.')

if __name__ == '__main__':
    sys.exit(main())