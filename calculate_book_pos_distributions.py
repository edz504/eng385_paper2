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
    word_tags = nltk.pos_tag(tokens)
    cfd = nltk.ConditionalFreqDist(word_tags)
    fd = nltk.FreqDist(tag for (word, tag) in word_tags)
    title_to_freq[title] = (cfd, fd)

with open('data/title_to_freq.pickle', 'wb') as f:
    pickle.dump(title_to_freq, f)

# print cfd['run'].most_common()