import pandas as pd

dictCSV=pd.read_csv("data/character_dictionary.csv", encoding="utf-8")
pingyum = {}

for i, x in enumerate(dictCSV["x"]):
    pingyum[x] = dictCSV["y"][i]

def romanize(input_word):
    i=0
    output_word=""
    while i < len(input_word):
        if input_word[i] in pingyum:
            output_word+=pingyum[input_word[i]]
        else:
            output_word+=input_word[i]
        i+=1
    return output_word
