# Overview of Methodology
###### This is written as a precursor to the reflection paper, and in order to rigorously document the process.

### Cleaning
We read in the `book-tag-all.csv` as a `pandas` DataFrame and apply standard NLP cleaning processes (normalization) through trailing whitespace stripping, non-alphabet character removal, whitespace-to-underscore, and lowercasing.  We then lemmatize using the WordNet Lemmatizer from the `nltk` module.  The new DataFrame is then `pickle`'d after the new transformed columns are added.  This code is available in `clean.py`.

### Analysis
A brief analysis is conducted to explore the feasibility of various top-level categories.  We use Wordnet to investigate hypernyms and depths, ultimately coming to the conclusion in the section below.  This code is available in `analyze.py`.

### Categories
1.  Phrase (multiple words)
  * 2-word
  * 3-word
  * 4-word
2.  Letter (single alphabet letter)
  * thirds of the alphabet (a little meaningless)
3.  Other (no Wordnet synsets)
  * real word with no synset
  * jargon
  * typo
4.  Positive Adjective
  * 3 subclusters
5.  Negative Adjectives
  * 3 subclusters
6.  Other Adjectives
  * 3 subclusters
7.  Noun Abstraction (relying on Wordnet structure, entity hypernym ancestor)
  * 3 subclusters
8.  Noun Physical entity (same as above)
  * 3 subclusters
9.  Noun Thing (same as above)
  * 3 subclusters
10. Verb
  * 3 subclusters

### Steps and Notes by Top-Level Category
We first make a pass through every tag, classifying it into one of the top-level categories.  Then, we apply the further refinement detailed into one of the 3 subcategories, which can be an explicit set of cases or our modified K-Means clustering (see Clustering).  This pass is implemented in `categorize_top.py`.

1.  Check for '_'
  * check for # of '_'
2.  Check length
  * check alphabet placement
3.  Check if tag has any Wordnet synsets
  * check with enchant dictionary to see if it is in the dictionary but simply has no Wordnet synsets
  * check the jargon file to see if it's slang / jargon
  * if it is in neither, it is a typo.
    - [PyEnchant](http://stackoverflow.com/questions/3788870/how-to-check-if-a-word-is-an-english-word-with-python)
    - [Slang list](http://onlineslangdictionary.com/word-list/0-a/), extraction is done in `extract_slang.py`

    Now, we must associate a tag to a part of speech and a synset in order to move forward.  For example, "run" can be a noun or a verb.  We use the text transcriptions of the ABC books to form a simple frequency distribution of adjective / verb / noun for each book.  Then, for each tag, we can collect its synsets and sample from the probability distribution associated with that book to select one of the synsets (always use the 01 synset, as they are ordered in vague usage ranking).  We do this once as we iterate through the tags.  The calculation of probability distributions and relevant wrangling is done in `calculate_book_pos_distributions.py`, and the selection of synsets is done in the pass through tags in `categorize_top.py`.  **Note that this results in a repeated tag for the same book having potentially different categorizations, which is acceptable from a theoretical standpoint.**

4.  Use an [external sentiment dataset](https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/blob/master/data/opinion-lexicon-English/negative-words.txt), which must be cleaned and lemmatized just like our tags
  * Apply clustering
5.  See #4.
6.  See #4.
7.  Check the lowest common hypernym of the noun with the abstraction.n.06 synset: if it is the abstraction.n.06 synset itself, then label this tag as an Noun Abstraction.
  * Apply clustering
8.  Same as #6, but with physical_entity.n.01.
9.  Else case for #7 and #8.
10. There's only one verb category.
  * Apply clustering


### Clustering (Subcategories)
Clustering is done with a modified k-means using Wu-Palmer similarity scores, with the following steps:

  1.  Pick k random synsets among the N synsets in our subset.
  2.  Iterate through all N synsets, assigning them to the cluster with the largest similarity score (Wu-Palmer is [0, 1]).
  3.  For each of the k clusters, pick a new center.  This is done by selecting the synset that maximizes the within-cluster similarity score, and is akin in normal k-means to calculating the Euclidean mean.
  4.  Repeat steps 2-3 until convergence.
  5.  Examine cluster sizes to ensure some degree of reasonable-ness.
  6.  Try to identify cluster characteristics in order to apply sub-labels.


