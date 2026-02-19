import argparse
import os
import sys
import urllib.request
import urllib.parse

BASE = 'https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg'

# A few well-known targets (J2000). Values are hard-coded for teaching.
TARGETS = [
  ('01.jpg', 'M51_field', 202.467917, 47.198333, 0.25, 512, 512, ''),
  ('02.jpg', 'M87_field', 187.705833, 12.391111, 0.25, 512, 512, ''),
  ('03.jpg', 'M101_field',210.804167, 54.348056, 0.25, 512, 512, ''),
  ('04.jpg', 'SDSS_field',224.594100, -1.090000, 0.25, 512, 512, ''),
]

def fetch(url, outpath):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    with open(outpath,'wb') as f:
        f.write(data)

def add_white_border_jpeg(inpath, outpath, border=35):
    # Keep dependencies minimal: we do a crude border by rewriting bytes into PIL if available.
    try:
        from PIL import Image, ImageOps
        img = Image.open(inpath)
        img2 = ImageOps.expand(img, border=border, fill='white')
        img2.save(outpath, quality=95)
    except Exception as e:
        # If PIL isn't available, just copy the original.
        # The MATLAB script will still run; the border-crop demo will just be less dramatic.
        if inpath != outpath:
            with open(inpath,'rb') as f: data = f.read()
            with open(outpath,'wb') as f: f.write(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for fname, name, ra, dec, scale, w, h, opt in TARGETS:
        params = dict(ra=ra, dec=dec, scale=scale, width=w, height=h, opt=opt)
        url = BASE + '?' + urllib.parse.urlencode(params)
        outpath = os.path.join(args.out, fname)
        tmp = outpath + '.tmp'
        print(f'Downloading {name}: {url}')
        fetch(url, tmp)
        if fname == '02.jpg':
            add_white_border_jpeg(tmp, outpath, border=45)
            os.remove(tmp)
        else:
            os.replace(tmp, outpath)

if __name__ == '__main__':
    sys.exit(main())