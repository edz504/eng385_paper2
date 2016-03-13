import cPickle as pickle
from fuzzywuzzy import process
import nltk
import numpy as np
import os
import pandas as pd
import re

with open('data/cleaned_tags.pickle', 'rb') as f:
    tag_df = pickle.load(f)

# Use fuzzy matching to create dictionary.
book_titles = set(tag_df.Book)
filename_to_dict = {}
for filename in os.listdir(
    os.path.join(os.getcwd(), 'data/abc-all')):
    # Ignore titles file
    if filename == '__titles.txt':
        continue
    # Remove .txt 
    s = filename[:-4]

    # Keep letters only
    cleaned_fn = ''.join([i for i in s if i.isalpha()])
    filename_to_dict[filename] = process.extractOne(cleaned_fn,
                                                    book_titles)[0]

# Manually inspect, then fix
filename_to_dict['4844136-an-alphabet.txt'] = 'ABC An Alphabet'
filename_to_dict['scriptures2.txt'] = "The Children's Moral Alphabet"

for book_title in book_titles:
    if book_title not in filename_to_dict.values():
        print book_title

# First implement a function that takes in a FreqDist
# and creates our own dictionary (Object def not necessary)
# that combines
# NN, NNP, NNPS, NNS into 'n',
# VB, VBD, VBG, VBN, VBP, VBZ into 'v'
# JJ, JJR, JJS into 'a'
# in order to facilitate the connection between the pos
# that Wordnet's single tags produce and the nltk pos_tag
# terms.  After combining the counts (and ignoring terms
# that don't fall into the above categories), we normalize
# to produce probabilities.
def convert_pos(fd):
    n = 0
    v = 0
    a = 0
    for key, value in fd.iteritems():
        if key in ['NN', 'NNP', 'NNPS', 'NNS']:
            n += value
        elif key in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            v += value
        elif key in ['JJ', 'JJR', 'JJS']:
            a += value
    total = float(n + v + a)
    if total == 0:
        return None
    return {'n': n / total,
            'v': v / total,
            'a': a / total}


# Go through each file and create a dictionary that has
# key = book title, value = (CFD, FD).  We use the
# CFD if the tag in question is in the book's transcription --
# if it is not, we use the FD to find the highest frequency
# POS.
title_to_freq = {}
for fn, title in filename_to_dict.iteritems():
    with open(os.path.join('data/abc-all', fn), 'rb') as f:
        raw_text = f.read()
    tokens = nltk.word_tokenize(raw_text.decode('utf-8'))
    # We need to apply the cleaning we used to clean our own
    # tags in order to check the tags later, but we avoid
    # lemmatizing to preserve contextual tagging.
    cleaned_tokens = [re.sub(r"\s+", '_',
                             re.sub("[^a-zA-z\s+]+",
                                    '',
                                    s.lower()).strip())
                      for s in tokens]

    word_tags = nltk.pos_tag([t for t in cleaned_tokens if len(t) > 0])
    cfd = nltk.ConditionalFreqDist(word_tags)
    all_fd = nltk.FreqDist(tag for (word, tag) in word_tags)
    
    # Now, we turn the cfd and fd into n/v/a probability
    # distributions with the convert_pos function above.
    cpd = dict([(token, convert_pos(fd))
                 for (token, fd) in cfd.iteritems()])
    title_to_freq[title] = (cpd, convert_pos(all_fd))

with open('data/title_to_freq.pickle', 'wb') as f:
    pickle.dump(title_to_freq, f)
