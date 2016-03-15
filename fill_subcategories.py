import cPickle as pickle
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd

# Fill in the clustering results (positive adjectives,
# negative adjectives, other adjectives, noun abstractions,
# noun physical entities, adverbs, verbs)
with open('data/tag_category1.pickle', 'rb') as f:
    tag_df = pickle.load(f)

# Keep a pointer for each of the clustered top categories.
# As we move down the sub_category, we increment the corresponding
# pointers.
index = [0] * 7
subcategory_assignments = [None] * 7

with open('data/adj_pos_clustering.pickle', 'rb') as f:
    subcategory_assignments[0] = pickle.load(f)[1]
with open('data/adj_neg_clustering.pickle', 'rb') as f:
    subcategory_assignments[1] = pickle.load(f)[1]
with open('data/adj_oth_clustering.pickle', 'rb') as f:
    subcategory_assignments[2] = pickle.load(f)[1]
with open('data/noun_abstraction_clustering.pickle', 'rb') as f:
    subcategory_assignments[3] = pickle.load(f)[1]
with open('data/noun_physicalentity_clustering.pickle', 'rb') as f:
    subcategory_assignments[4] = pickle.load(f)[1]
with open('data/adverb_clustering.data', 'rb') as f:
    subcategory_assignments[5] = f.read().split('\n')[:-1]
with open('data/verb_clustering.pickle', 'rb') as f:
    subcategory_assignments[6] = pickle.load(f)[1]

subcategory_fill = [None] * len(tag_df)
for i, subcategory in enumerate(tag_df['Subcategory']):
    if np.isnan(subcategory):
        category = tag_df['Top Category'][i]
        category_i = category - 4
        these_assignments = subcategory_assignments[category_i]
        # Add one to cluster values
        subcategory_fill[i] = int(these_assignments[index[category_i]]) + 1
        index[category_i] += 1
    else:
        subcategory_fill[i] = tag_df['Subcategory'][i]

# Check that all pointers are at their ends
for i in xrange(0, 7):
    assert index[i] == len(subcategory_assignments[i])

tag_df['Subcategory (Final)'] = [int(i)
                                 for i in subcategory_fill]

with open('data/tag_category2.pickle', 'wb') as f:
    pickle.dump(tag_df, f)

tag_df['Unique Category'] = (tag_df['Top Category'].map(str) +
                             "-" +
                             tag_df['Subcategory (Final)'].map(str))

tag_df.to_csv('data/tag_categorized.csv',
              index=False)