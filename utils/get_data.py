import slumber
import requests
import tqdm
from slumber.exceptions import HttpNotFoundError
import numpy as np

app_id = <YOUR API ID>
app_key = <YOUR APP KEY>
s = requests.session()
s.headers["app_id"] = app_id
s.headers["app_key"] = app_key
s.headers["Accept"] = "application/json"
api = slumber.API(
    "https://od-api.oxforddictionaries.com:443/api/v1",
    session=s,
    append_slash=False
)

data = open("./extracted_lemmas.txt", "r").read().splitlines()


def get_word_info(word):
    try:
        definitions = api.entries.en(word).definitions.get()["results"][0]
        lemma = word
    except HttpNotFoundError as e:
        try:
            r = api.inflections.en(word).get()["results"][0]
            new_word = r["lexicalEntries"][0]["inflectionOf"][0]["id"]
            if new_word != word:
                definitions = api.entries.en(new_word).definitions.get()
                definitions = definitions["results"][0]
                lemma = new_word
            else:
                return False, "No such word in Oxford Dictionaries!"
        except HttpNotFoundError:
            return False, "No such word in Oxford Dictionaries!"

    try:
        sentences = api.entries.en(lemma).sentences.get()["results"][0]
    except HttpNotFoundError as e:
        return False, "No sentence examples in Oxford Dictionaries!"

    id2sentences = {}

    n = len(sentences["lexicalEntries"])
    for i in range(n):
        m = len(sentences["lexicalEntries"][i]["sentences"])
        for j in range(m):
            num_senses = sentences["lexicalEntries"][i]["sentences"][j]
            num_senses = len(num_senses["senseIds"])
            sentence = sentences["lexicalEntries"][i]["sentences"][j]["text"]
            for k in range(num_senses):
                senseId = sentences["lexicalEntries"][i]["sentences"][j]
                senseId = senseId["senseIds"][k]
                if senseId in id2sentences:
                    id2sentences[senseId].append(sentence)
                else:
                    id2sentences[senseId] = [sentence, ]

    n = len(definitions["lexicalEntries"])
    for i in range(n):
        m = len(definitions["lexicalEntries"][i]["entries"])
        for j in range(m):
            num_senses = definitions["lexicalEntries"][i]["entries"][j]
            num_senses = len(num_senses["senses"])
            for k in range(num_senses):
                ij = definitions["lexicalEntries"][i]["entries"][j]
                ijk = ij["senses"][k]
                if "definitions" in ijk.keys():
                    definition = ijk["definitions"][0]
                    senseId = ijk["id"]
                    if senseId in id2sentences.keys():
                        np.random.seed(42)
                        return True, lemma, definition, \
                            np.random.choice(id2sentences[senseId])
                if "subsenses" in ijk.keys():
                    num_subsenses = len(ijk["subsenses"])
                    for l in range(num_subsenses):
                        definition = ijk["subsenses"][l]["definitions"][0]
                        senseId = ijk["subsenses"][l]["id"]
                        if senseId in id2sentences.keys():
                            np.random.seed(42)
                            return True, lemma, definition, \
                                np.random.choice(id2sentences[senseId])

    return False, "Info not found for word in Oxford Dictionaries!"

for i in range(len(data[:100])):
    print(get_word_info(data[i]))
