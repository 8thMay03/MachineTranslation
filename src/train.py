import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load
from dataset import build_dataset

def preprocess_function(examples, tokenizer, src_lang="en", tgt_lang="vi", max_source_length=256, max_target_length=256):
    # Mỗi examples["translation"] là list chứa các dict {en, vi}
    translations = examples["translation"]
    
    # Trích xuất câu nguồn và đích
    inputs = [t[src_lang] for t in translations]
    targets = [t[tgt_lang] for t in translations]

    # Tokenize câu nguồn
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding="max_length"
    )

    # Tokenize câu đích (label)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

class Args:
    train_src = r"D:\Datasets\PhoMT\detokenization\train\train.en"
    train_tgt = r"D:\Datasets\PhoMT\detokenization\train\train.vi"
    val_src = r"D:\Datasets\PhoMT\detokenization\dev\dev.en"
    val_tgt = r"D:\Datasets\PhoMT\detokenization\dev\dev.vi"
    output_dir = "../checkpoints"
    model_name = "../checkpoints/checkpoint-49995"
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    num_train_epochs = 1
    learning_rate = 5e-5

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train_src", required=True)
    # parser.add_argument("--train_tgt", required=True)
    # parser.add_argument("--val_src", required=True)
    # parser.add_argument("--val_tgt", required=True)
    # parser.add_argument("--output_dir", default="./out")
    # parser.add_argument("--model_name", default="Helsinki-NLP/opus-mt-en-vi")
    # parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    # parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    # parser.add_argument("--num_train_epochs", type=int, default=3)
    # parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = Args()

    ds = build_dataset(args.train_src, args.train_tgt, args.val_src, args.val_tgt)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized = ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True, remove_columns=ds["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = load("sacrebleu")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # replace -100 in the labels as we can't decode them
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # sacrebleu expects list of references for each pred
        decoded_labels = [[l] for l in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=None,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=True if os.getenv("USE_FP16", "1")=="1" else False,
        push_to_hub=False,
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
