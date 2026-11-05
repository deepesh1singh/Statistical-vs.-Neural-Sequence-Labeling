"""Downloads glove.6B.zip and extracts only glove.6B.100d.txt into ./glove/"""
import argparse
import os
import sys
import urllib.request
import zipfile

URLS = [
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
]

def hook(blocks, bsize, total):
    if total > 0:
        pct = min(100.0, blocks * bsize * 100.0 / total)
        sys.stdout.write("\r  %.1f%%" % pct)
        sys.stdout.flush()

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(here, "glove"))
    ap.add_argument("--keep_zip", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    target = os.path.join(args.out_dir, "glove.6B.100d.txt")
    if os.path.isfile(target):
        print("already exists:", target)
        return
    zpath = os.path.join(args.out_dir, "glove.6B.zip")
    ok = False
    for url in URLS:
        try:
            print("downloading", url)
            urllib.request.urlretrieve(url, zpath, hook)
            print()
            ok = True
            break
        except Exception as ex:
            print("\nfailed: %s: %s" % (type(ex).__name__, ex))
    if not ok:
        print("Manual: download glove.6B.zip, extract glove.6B.100d.txt to", target)
        sys.exit(1)
    with zipfile.ZipFile(zpath) as z:
        z.extract("glove.6B.100d.txt", args.out_dir)
    if not args.keep_zip:
        os.remove(zpath)
    print("ready:", target)

if __name__ == "__main__":
    main()