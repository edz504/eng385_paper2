from collections import Counter
import cPickle as pickle
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd

# Implemental definitional scoring
def definitional_score(synset1, synset2):
    keynouns1 = [t[0] for t in
                 nltk.pos_tag(
                     nltk.word_tokenize(
                         synset1.definition()))
                 if t[1] in ['NN', 'NNP', 'NNPS', 'NNS']]
    synset_nouns1 = []
    for keynoun in keynouns1:
        all_synset_nouns = [s for s in wn.synsets(keynoun)
                            if s.pos() == 'n']
        if len(all_synset_nouns) > 0:
            synset_nouns1.append(all_synset_nouns[0])
    keynouns2 = [t[0] for t in
                 nltk.pos_tag(
                     nltk.word_tokenize(
                         synset2.definition()))
                 if t[1] in ['NN', 'NNP', 'NNPS', 'NNS']]
    synset_nouns2 = []
    for keynoun in keynouns2:
        all_synset_nouns = [s for s in wn.synsets(keynoun)
                            if s.pos() == 'n']
        if len(all_synset_nouns) > 0:
            synset_nouns2.append(all_synset_nouns[0])
    score = 0.
    for synset_noun1 in synset_nouns1:
        for synset_noun2 in synset_nouns2:
            score += wn.wup_similarity(synset_noun1,
                                       synset_noun2)
    # Note to normalize by size of synset_noun set (both)
    if (len(synset_nouns1) + len(synset_nouns2)) != 0:
        return score / (len(synset_nouns1) + len(synset_nouns2))
    else:
        return 0.

# Implement our clustering heuristic
def cluster(k, synset_list, definitional, ITER):
    # Inputs:
    # k, an integer number of clusters
    # synset_list, a list of Synset objects
    # definitional: a boolean indicating whether to use
    #               direct Wu-Palmer or the wup of the
    #               definition
    # ITER: number of random initializations

    # The imbalance list has tuple where
    # tuple[0] = centers (numpy array of synset centers)
    # tuple[1] = cluster_assignments
    # tuple[2] = mae from even split
    # tuple[3] = sum of within-cluster similarity
    imbalance = [None] * ITER
    for iter in xrange(ITER):
        cluster_assign = [None] * len(synset_list)
        swcs = 0
        center_inds = np.random.choice(len(synset_list),
                                       k, replace=False)
        centers = np.array(synset_list)[center_inds]

        # Assign each point to the best cluster
        for i, synset in enumerate(synset_list):
            if not definitional:
                scores = [wn.wup_similarity(center,
                                            synset)
                          for center in centers]
            else:
                scores = [definitional_score(center,
                                             synset)
                          for center in centers]

            max_center_inds = [c for c, j in enumerate(scores)
                               if j == max(scores)]
            cluster_assign[i] = np.random.choice(max_center_inds, 1)[0]
            swcs += scores[cluster_assign[i]]
        count = Counter(cluster_assign)
        mae = np.mean(abs(np.array(count.values()) -
                          sum(count.values()) * 1. / k))
        imbalance[iter] = (centers,
                           cluster_assign,
                           mae,
                           swcs)
        if iter % 10 == 0:
            print '{0} / {1} initializations done'.format(
                iter, ITER)

    return imbalance
        
with open('data/tag_category1.pickle', 'rb') as f:
    tag_df = pickle.load(f)

#######
# Verbs have a large number of short hierarchies, so Wu-Palmer
# will work but not as well
verb_list = [wn.synset(x)
             for x in tag_df['Selected Synset']
             if (x is not None and
                 wn.synset(x).pos() == 'v')]
imb = cluster(3, verb_list, False, 100)

smallest_mae = sorted(imb, key=lambda x: x[2])
largest_sim = sorted(imb, key=lambda x : x[3],
                     reverse=True)
# The clustering with the most even split has a
# 4509 / 4504 / 4490 split (perfectly even split is
# 4501 / 4501 / 4501).  On average, the Wu-Palmer
# similarity of a synset to its center is 0.2288.

# The clustering with the largest aggregate similarity
# has an average Wu-Palmer similarity of 0.3644.  It
# has a split of 12355, 935, 213, though.

# We choose the most even split clustering, and save it.
with open('data/verb_clustering.pickle', 'wb') as f:
    clustering = ([s.name() for s in smallest_mae[0][0]],
                  smallest_mae[0][1])
    pickle.dump(clustering, f)

######
# Nouns have a hierarchy, so they can have shortest_paths,
# and Wu-Palmer works well. Our clustering takes much longer
# because the noun hierarchy is much larger.

## Category 7 (Noun Abstraction)
noun_list = [wn.synset(s)
             for s, cat
             in zip(tag_df['Selected Synset'],
                    tag_df['Top Category'])
             if (s is not None and
                 cat == 7)]
imb = cluster(3, noun_list, False, 100)

smallest_mae = sorted(imb, key=lambda x: x[2])
largest_sim = sorted(imb, key=lambda x : x[3],
                     reverse=True)

# We choose the most even split clustering again (it
# has 0.3406 aggregate Wu-Palmer similarity), and save it.
with open('data/noun_abstraction_clustering.pickle', 'wb') as f:
    clustering = ([s.name() for s in smallest_mae[0][0]],
                  smallest_mae[0][1])
    pickle.dump(clustering, f)

## Category 8 (Noun Physical_Entity)
noun_list = [wn.synset(s)
             for s, cat
             in zip(tag_df['Selected Synset'],
                    tag_df['Top Category'])
             if (s is not None and
                 cat == 8)]
imb = cluster(3, noun_list, False, 100)

smallest_mae = sorted(imb, key=lambda x: x[2])
largest_sim = sorted(imb, key=lambda x : x[3],
                     reverse=True)

# We choose the most even split clustering again (it
# has 0.5287 aggregate Wu-Palmer similarity), and save it.
with open('data/noun_physicalentity_clustering.pickle', 'wb') as f:
    clustering = ([s.name() for s in smallest_mae[0][0]],
                  smallest_mae[0][1])
    pickle.dump(clustering, f)


######
# Adjectives require a different model.  Possibly use
# some comparison between the definitions?  We could tokenize
# the definitions and find all nouns within the
# definition, and then aggregate similarity.

## Category 4 (Positive Adjectives)
adj_list = [wn.synset(s)
            for s, cat
            in zip(tag_df['Selected Synset'],
                   tag_df['Top Category'])
            if (s is not None and
                cat == 4)]
imb = cluster(3, adj_list, True, 100)

smallest_mae = sorted(imb, key=lambda x: x[2])
largest_sim = sorted(imb, key=lambda x : x[3],
                     reverse=True)

# We choose the most even split clustering again (it
# has 0.3325 aggregate Wu-Palmer similarity), and save it.
with open('data/adj_pos_clustering.pickle', 'wb') as f:
    clustering = ([s.name() for s in smallest_mae[0][0]],
                  smallest_mae[0][1])
    pickle.dump(clustering, f)

## Category 5 (Negative Adjectives)
adj_list = [wn.synset(s)
            for s, cat
            in zip(tag_df['Selected Synset'],
                   tag_df['Top Category'])
            if (s is not None and
                cat == 5)]
imb = cluster(3, adj_list, True, 100)

smallest_mae = sorted(imb, key=lambda x: x[2])
largest_sim = sorted(imb, key=lambda x : x[3],
                     reverse=True)

with open('data/adj_neg_clustering.pickle', 'wb') as f:
    clustering = ([s.name() for s in smallest_mae[0][0]],
                  smallest_mae[0][1])
    pickle.dump(clustering, f)

## Category 5 (Other Adjectives)
adj_list = [wn.synset(s)
            for s, cat
            in zip(tag_df['Selected Synset'],
                   tag_df['Top Category'])
            if (s is not None and
                cat == 6)]
imb = cluster(3, adj_list, True, 100)

smallest_mae = sorted(imb, key=lambda x: x[2])
largest_sim = sorted(imb, key=lambda x : x[3],
                     reverse=True)

with open('data/adj_oth_clustering.pickle', 'wb') as f:
    clustering = ([s.name() for s in smallest_mae[0][0]],
                  smallest_mae[0][1])
    pickle.dump(clustering, f)

######
# There are only 42 unique adverbs, so hand-categorization
# is viable, see adverb_clustering.txt

