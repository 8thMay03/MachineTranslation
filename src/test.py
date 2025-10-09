from inference import Translator

translator = Translator('../checkpoints')

text = "Ngl, you are handsome!"
translated = translator.translate(text)

print("Input:", text)
print("Output:", translated)