from inference import Translator

translator = Translator('../checkpoints/checkpoint-21000')

while True:
    text = input("Enter text: ")
    translated = translator.translate(text)

    print("Output:", translated)