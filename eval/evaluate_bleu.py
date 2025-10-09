from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

def translate_batch(model, tokenizer, texts, max_length=256):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outs = model.generate(**inputs, max_length=max_length, num_beams=5)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outs]

def evaluate(model_dir, src_file, ref_file):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    with open(src_file, encoding='utf-8') as f:
        src = [l.strip() for l in f]
    with open(ref_file, encoding='utf-8') as f:
        ref = [l.strip() for l in f]
    preds = []
    batch = 32
    for i in range(0, len(src), batch):
        preds.extend(translate_batch(model, tokenizer, src[i:i+batch]))
    bleu = sacrebleu.corpus_bleu(preds, [ref])
    print("BLEU:", bleu.score)
    return bleu.score

if __name__ == "__main__":
    import sys
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
