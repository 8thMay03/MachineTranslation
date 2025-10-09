from datasets import load_dataset, Dataset, DatasetDict
import os

def load_parallel_text(src_file, tgt_file, src_lang="en", tgt_lang="vi"):
    with open(src_file, encoding='utf-8') as f:
        src_lines = [l.strip() for l in f.readlines()]
    with open(tgt_file, encoding='utf-8') as f:
        tgt_lines = [l.strip() for l in f.readlines()]
    assert len(src_lines) == len(tgt_lines)
    data = {"translation": [{src_lang: s, tgt_lang: t} for s, t in zip(src_lines, tgt_lines)]}
    return Dataset.from_dict(data)

def build_dataset(train_src, train_tgt, val_src, val_tgt, src_lang="en", tgt_lang="vi"):
    train = load_parallel_text(train_src, train_tgt, src_lang, tgt_lang)
    val = load_parallel_text(val_src, val_tgt, src_lang, tgt_lang)
    return DatasetDict({"train": train, "validation": val})

# ds = build_dataset('../data/raw/train.en', '../data/raw/train.vi', '../data/raw/validation.en', '../data/raw/validation.vi')
# print(ds['train'])