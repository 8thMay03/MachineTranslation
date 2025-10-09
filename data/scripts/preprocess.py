# Mục tiêu: hợp nhất cặp file, clean cơ bản, loại bỏ cặp rỗng và length outliers
import argparse
import re

def clean_line(s):
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def filter_pairs(src_path, tgt_path, out_src, out_tgt, min_len=1, max_len=200):
    with open(src_path, 'r', encoding='utf-8') as fs, open(tgt_path, 'r', encoding='utf-8') as ft:
        src_lines = fs.readlines()
        tgt_lines = ft.readlines()

    assert len(src_lines) == len(tgt_lines), "Số dòng không khớp"

    kept_src = []
    kept_tgt = []
    for s, t in zip(src_lines, tgt_lines):
        s = clean_line(s)
        t = clean_line(t)
        if not s or not t:
            continue
        ls, lt = len(s.split()), len(t.split())
        if not (min_len <= ls <= max_len and min_len <= lt <= max_len):
            continue
        kept_src.append(s + "\n")
        kept_tgt.append(t + "\n")

    with open(out_src, 'w', encoding='utf-8') as osf:
        osf.writelines(kept_src)
    with open(out_tgt, 'w', encoding='utf-8') as otf:
        otf.writelines(kept_tgt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--out_src", required=True)
    p.add_argument("--out_tgt", required=True)
    p.add_argument("--min_len", type=int, default=1)
    p.add_argument("--max_len", type=int, default=200)
    args = p.parse_args()
    filter_pairs(args.src, args.tgt, args.out_src, args.out_tgt, args.min_len, args.max_len)
