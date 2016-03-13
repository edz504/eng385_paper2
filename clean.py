import cPickle as pickle
import nltk
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# 54787 tags
tag_df = pd.read_csv('data/book-tag-all.csv')
tags = tag_df['Tag']
print len(tags)

print len(set(tags))
# 9226 unique tags


# Lowercase, strip whitespace from beginning and end,
# replace internal whitespace with underscores,
# remove all non-alphabet characters (numbers, punctuation)
cleaned_tags = [re.sub(r"\s+", '_',
                       re.sub("[^a-zA-z\s+]+",
                       '',
                       t.lower().strip()))
                for t in tags]
tag_df['Cleaned_Tag'] = cleaned_tags
print len(set(cleaned_tags))
# 9151 post-cleaning


# # Stemming -- computationally quicker, but lemma
# # is preferable due to higher level of sophistication.
# porter_stemmer = PorterStemmer()

# stemmed_tags = [porter_stemmer.stem(t)
#                 for t in cleaned_tags]
# tag_df['Stemmed_Tag'] = stemmed_tags
# print len(set(stemmed_tags))
# # 7517 post-stemming


# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
lemma_tags = [wordnet_lemmatizer.lemmatize(t)
              for t in cleaned_tags]
tag_df['Lemmatized_Tag'] = lemma_tags
print len(set(lemma_tags))
# 8455 unique lemmatized_tags
print len(set([t for t in lemma_tags if '_' in t]))
# 1181 of which are multi_word

with open('cleaned_tags.pickle', 'wb') as f:
    pickle.dump(tag_df, f)