from datasets import load_dataset

dataset = load_dataset("thainq107/iwslt2015-en-vi")
for split in ["train", "validation", "test"]:
    with open(f"data/raw/{split}.en", "w", encoding="utf-8") as f_en, \
         open(f"data/raw/{split}.vi", "w", encoding="utf-8") as f_vi:
        for item in dataset[split]:
            f_en.write(item["en"].strip() + "\n")
            f_vi.write(item["vi"].strip() + "\n")
