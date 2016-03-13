import cPickle as pickle
import enchant
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

with open('data/cleaned_tags.pickle', 'rb') as f:
    tag_df = pickle.load(f)

with open('data/title_to_freq.pickle', 'rb') as f:
    title_to_freq = pickle.load(f)

# Fill this in 1-10
top_category = [None] * len(tag_df)

# For non-clustering subcategories, fill this in 1-3.
# For clustering subcategories, leave as None and fill
# in later.
sub_category = [None] * len(tag_df)

# Initialize dictionary for checking English words.
d = enchant.Dict("en_US")

for i, row in enumerate(tag_df.itertuples()):
    tag = row[7]

    # Check category 1 (and classify subcategory)
    spl = tag.split('_')
    if len(spl) > 1:
        top_category[i] = 1
        if len(spl) == 2:
            sub_category[i] = 1
        elif len(spl) == 3:
            sub_category[i] = 2
        else:
            sub_category[i] = 3
        continue

    # Check category 2 (and classify subcategory)
    if len(tag) == 1:
        top_category[i] = 2
        if tag <= 'i':
            sub_category[i] = 1
        elif tag <= 'r':
            sub_category[i] = 2
        else:
            sub_category[i] = 3
        continue

    # Check category 3
    synsets = wn.synsets(tag)
    if len(synsets) == 0:
        top_category[i] = 3
        if d.check(tag):
            sub_category[i] = 1
        elif 
