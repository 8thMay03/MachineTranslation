from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, model_dir="./out", model_name=None):
        if model_name and not model_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    def translate(self, text, max_length=256, num_beams=5):
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

