import cPickle as pickle
import nltk
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

with open('data/cleaned_tags.pickle', 'rb') as f:
    tag_df = pickle.load(f)

lemma_tags = list(tag_df.Lemmatized_Tag)

# For each lemmatized tag, compute its synsets.  Then,
# calculate min_depth() and max_depth() for each synset,
# and average all these values to get a depth score.  If
# the tag is a phrase (contains '_'), then do this for
# all individual words and average over all words.    
def compute_depth_score(tag):
    if '_' in tag:
        depth_scores = [compute_depth_score_helper(t)
                        for t in tag.split('_')]
        if None in depth_scores:
            return None
        else:
            return np.mean(depth_scores)
    else:
        return compute_depth_score_helper(tag)

def compute_depth_score_helper(tag):
    synsets = wn.synsets(tag)
    if len(synsets) == 0:
        return None
    total = 0.
    for synset in synsets:
        total += (synset.min_depth() +
                  synset.max_depth())
    return total / (2 * len(synsets))

depth_scores = [compute_depth_score(t) for t in lemma_tags]
tag_df['Depth_Score'] = depth_scores

# See unique sorted
scored_tags = zip(lemma_tags, depth_scores)
scored_tags_unique = list(set(scored_tags))
sorted_scored_tags = sorted(
    scored_tags_unique,
    key=lambda x: (x[1] is None, x[1]))
print sorted_scored_tags[0:10]


# The root depth scoring has led us to understand Wordnet
# better -- there is different organization in Wordnet
# for adjectives, nouns, and verbs.

# Adjectives have "similar to", whereas noun and verbs
# have real IsA and KindOf relations.  We want to look
# at the top-level of verbs and nouns.

# Noun root = 'entity.n.01', with three hyponyms:
# abstraction.n.06, physical_entity.n.01, thing.n.08
# ^ From documentation (https://wordnet.princeton.edu/)

# Verb
all_root_verbs = []
for tag in lemma_tags:
    synsets = wn.synsets(tag)
    for synset in synsets:
        if synset.pos() == 'v':
            all_root_verbs += synset.root_hypernyms()

unique_root_verbs = set(all_root_verbs)
print len(unique_root_verbs)
# 430 unique root verbs