from bs4 import BeautifulSoup
import re
import requests
from nltk.stem import WordNetLemmatizer

all_slang = []
page_endings = ['0-a', 'b-b', 'c-d', 'e-f', 'g-g', 'h-k',
                'l-o', 'p-r', 's-s', 't-t', 'u-w', 'x-z']
for page_ending in page_endings:
    url = 'http://onlineslangdictionary.com/word-list/{0}/'.format(
        page_ending)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    soup_text = soup.get_text()
    start = soup_text.find('Words in bold are Featured Words.') + 38
    end = soup_text.find('Click here to show variants') - 2
    all_slang += soup_text[start:end].split('\n')

# We need to apply the cleaning we used to clean our own
# tags in order to have a chance to match a slang word
cleaned_slang = [re.sub(r"\s+", '_',
                        re.sub("[^a-zA-z\s+]+",
                               '',
                               s.lower()).strip())
                 for s in all_slang]

wordnet_lemmatizer = WordNetLemmatizer()
lemma_slang = [wordnet_lemmatizer.lemmatize(t)
               for t in cleaned_slang]

# Write to file, removing now-blank slang
with open('data/slang.txt', 'wb') as f:
    for s in lemma_slang:
        if len(s) > 0:
            f.write(s + '\n')