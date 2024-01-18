from googletrans import Translator

text1 = "Once upon a time, there was a Greek King, Midas. He was very rich and had lots of Gold. He had a daughter, who he loved a lot. One day, Midas found an angel in need of help. He helped her and in return she agreed to grant a wish. "

translator = Translator()

print("Translate English to ESP : ", translator.translate(text1, src='en', dest='ml'))