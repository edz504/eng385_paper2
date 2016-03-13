import cPickle as pickle
import enchant
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd

# Our cleaned tag DataFrame
with open('data/cleaned_tags.pickle', 'rb') as f:
    tag_df = pickle.load(f)

# Our frequency distribution data
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

# Load slang list.
with open('data/slang.txt', 'rb') as f:
    slang_list = f.read().split('\n')

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

    # Check category 3 (and classify subcategory)
    synsets = wn.synsets(tag)
    if len(synsets) == 0:
        top_category[i] = 3
        if d.check(tag):
            sub_category[i] = 1
        elif tag in slang_list:
            sub_category[i] = 2
        else:
            sub_category[i] = 3

    # Associate the tag to a specific synset / POS if
    # it is not category 1, 2, or 3.
    title = row[1]
    cpd, pd = title_to_freq[title]
    if tag in cpd: # Check if the tag is in the CPD
        this_pd = cpd[tag]
    else: # Use the overall distribution of POS in the book
        this_pd = pd
    # Take the pos' given from the synsets and re-normalize
    # the distribution with only the pos given by the synsets.
    synset_pos = set(synset.pos() for synset in synsets)
    new_sum = sum([v for k, v in this_pd.iteritems() if k in synset_pos])
    new_pd = dict((k, v / new_sum)
                  for k, v in this_pd.iteritems() if k in synset_pos)

    rv = np.random.choice(new_pd.keys(), 1, new_pd.values())[0]

    if rv == 'a':
        pass
    elif rv == 'n':
        pass
    elif rv == 'v':
        pass

elements = np.array([1.1, 2.2, 3.3])
probabilities = np.array([0.2, 0.5, 0.3])
print np.random.choice(elements, 1, list(probabilities))