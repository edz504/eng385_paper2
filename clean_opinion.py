import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

wordnet_lemmatizer = WordNetLemmatizer()
with open('data/negative.txt', 'rb') as f:
    negative = f.read().split('\n')
cleaned_negative = [re.sub(r"\s+", '_',
                           re.sub("[^a-zA-z\s+]+",
                           '',
                           w.lower()).strip())
                    for w in negative]

lemma_neg = [wordnet_lemmatizer.lemmatize(w)
             for w in cleaned_negative]

with open('data/positive.txt', 'rb') as f:
    positive = f.read().split('\n')
cleaned_positive = [re.sub(r"\s+", '_',
                           re.sub("[^a-zA-z\s+]+",
                           '',
                           w.lower()).strip())
                    for w in positive]

lemma_pos = [wordnet_lemmatizer.lemmatize(w)
             for w in cleaned_positive]

with open('data/negative_clean.txt', 'wb') as f:
    for w in cleaned_negative:
        f.write(w + '\n')
with open('data/positive_clean.txt', 'wb') as f:
    for w in cleaned_positive:
        f.write(w + '\n')
