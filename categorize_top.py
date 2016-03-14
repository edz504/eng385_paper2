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

# Initialize dictionary for checking English words.
d = enchant.Dict("en_US")

# Load slang list.
with open('data/slang.txt', 'rb') as f:
    slang_list = f.read().split('\n')

# Load positive and negative lists.
with open('data/positive_clean.txt', 'rb') as f:
    positive = f.read().split('\n')

with open('data/negative_clean.txt', 'rb') as f:
    negative = f.read().split('\n')

# Define constants
ABSTR = wn.synset('abstraction.n.06')
PHYS = wn.synset('physical_entity.n.01')
TOL = 1e-5

# Record the selected synset if category 4-10
selected_synset = [None] * len(tag_df)

# Fill this in 1-10
top_category = [None] * len(tag_df)

# For non-clustering subcategories, fill this in 1-3.
# For clustering subcategories, leave as None and fill
# in later.
sub_category = [None] * len(tag_df)

for i, row in enumerate(tag_df.itertuples()):
    tag = row[7]
    synsets = wn.synsets(tag)

    spl = tag.split('_')
    if len(spl) > 1:              # Category 1
        top_category[i] = 1
        if len(spl) == 2:
            sub_category[i] = 1
        elif len(spl) == 3:
            sub_category[i] = 2
        else:
            sub_category[i] = 3
        continue

    if len(tag) == 1:           # Category 2
        top_category[i] = 2
        if tag <= 'i':
            sub_category[i] = 1
        elif tag <= 'r':
            sub_category[i] = 2
        else:
            sub_category[i] = 3
        continue

    if len(synsets) == 0:       # Category 3
        top_category[i] = 3
        # Enchant breaks if we feed it an empty string,
        # so label as typo (sub-category 3) if that happens.
        if len(tag) == 0:
            sub_category[i] = 3
        elif d.check(tag):
            sub_category[i] = 1
        elif tag in slang_list:
            sub_category[i] = 2
        else:
            sub_category[i] = 3
        continue

    # Associate the tag to a specific synset / POS if
    # it is not category 1, 2, or 3.
    title = row[1]
    cpd, pd = title_to_freq[title]
    synset_pos = [synset.pos() for synset in synsets]
    synset_pos_set = set(synset_pos)

    if tag in cpd: # Check if the tag is in the CPD
        if cpd[tag] is not None: # Note that some have values of None
            this_pd = cpd[tag]
        else:
            this_pd = pd
    else: # Use the overall distribution of POS in the book
        if pd is not None:
            this_pd = pd
        else: # Note that some overall distributions even have values of None
            this_pd = dict(zip(synset_pos_set,
                               [1. / len(synset_pos_set)
                                for s in synset_pos_set]))

    # Take the pos' given from the synsets and re-normalize
    # the distribution with only the pos given by the synsets.
    new_sum = sum([v for k, v in this_pd.iteritems() if k in synset_pos_set])
    if new_sum != 0:
        new_pd = dict((k, v / new_sum)
                      for k, v in this_pd.iteritems()
                      if k in synset_pos_set)
    else:
        if new_sum == 0:
            this_pd = pd
            new_sum =  sum([v for k, v in this_pd.iteritems()
                            if k in synset_pos_set])
            if new_sum == 0:
                new_pd = dict(zip(synset_pos_set,
                                   [1. / len(synset_pos_set)
                                    for s in synset_pos_set]))
            else:
                new_pd = dict((k, v / new_sum)
                              for k, v in this_pd.iteritems()
                              if k in synset_pos_set)

    # Verify that we have a normalized probability distribution
    assert sum(new_pd.values()) - 1 < TOL
    rv = np.random.choice(new_pd.keys(), 1, new_pd.values())[0]
    synset = synsets[synset_pos.index(rv)]

    selected_synset[i] = synset

    if rv == 'a' or rv == 's': # Category 4-6
        if tag in positive:
            top_category[i] = 4
        elif tag in negative:
            top_category[i] = 5
        else:
            top_category[i] = 6
        continue

    if rv == 'n':             # Category 7-9
        if synset.lowest_common_hypernyms(ABSTR)[0] == ABSTR:
            top_category[i] = 7
        elif synset.lowest_common_hypernyms(PHYS)[0] == PHYS:
            top_category[i] = 8
        continue

    # note most adverbs are lemmatized into adjectives
    # (cite Wordnet)
    if rv == 'r':
        top_category[i] = 9
    
    if rv == 'v':
        top_category[i] = 10

tag_df['Top Category'] = top_category
tag_df['Subcategory'] = sub_category
tag_df['Selected Synset'] = [s.name()
                             if s is not None
                             else None
                             for s in selected_synset]

with open('data/tag_category1.pickle', 'wb') as f:
    pickle.dump(tag_df, f)
