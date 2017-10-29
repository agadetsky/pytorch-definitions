from nltk import word_tokenize
from tqdm import tqdm

data = open("./lemma.en.txt").readlines()[10:]


def extract(sample):
    tokenized = word_tokenize(sample)
    if "/" in tokenized[0]:
        return tokenized[0].split("/")[0]
    else:
        return tokenized[0]



visited_words = set()
with open("./extracted_lemmas.txt", "w") as outfile:
    for i in tqdm(range(len(data))):
        extracted_word = extract(data[i])
        if extracted_word not in visited_words:
            outfile.write("{0}\n".format(extracted_word))
            visited_words.add(extracted_word)


