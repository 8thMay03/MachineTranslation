from inference import Translator

translator = Translator('../checkpoints')

while True:
    text = input("Enter text: ")
    translated = translator.translate(text)

    print("Output:", translated)