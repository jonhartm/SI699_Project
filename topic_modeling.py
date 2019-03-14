import os, re, string, sys, time

import pandas as pd
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

def print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()

print("loading stopwords...\n")
stopwords = nltk.corpus.stopwords.words('english')

start = time.time()
print("loading essays...")
orig_essays_df = pd.read_csv('data/opendata_essays000.gz', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])
stop = time.time()
print("({} s)\n".format(stop-start))

# downsample to start with
essays_df = orig_essays_df.sample(100)
# essays_df = orig_essays_df

# drop any rows where the essay field is blank
essays_df = essays_df[essays_df.essay.notnull()]

# regex to strip all punctuation and replace it with whitespace, then convert to lowercase and split on all whitespace
start = time.time()
print("creating unigrams...")
essays_df['unigrams'] = essays_df['essay'].apply(lambda x: re.sub(r'[^\w\s]|\r\n',' ',x).lower().split())
stop = time.time()
print("({} s)\n".format(stop-start))

# build bigram and trigram models with gensim
# set thresholds higher if we get too many
# from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
start = time.time()
print("building bigram/trigram models...")
bigram = gensim.models.Phrases([x for x in essays_df['unigrams']], min_count=500, threshold=10000)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram = gensim.models.Phrases(bigram[[x for x in essays_df['unigrams']]], threshold=10000)
trigram_mod = gensim.models.phrases.Phraser(trigram)
stop = time.time()
print("({} s)\n".format(stop-start))

# use the trigram and bigram models from above to get a list of tokens
# exclude any tokens that appear in the list "stopwords"
start = time.time()
print("creating tokens from unigram/bigram/trigrams...")
essays_df['tokens'] = essays_df['unigrams'].apply(lambda x: [token for token in trigram_mod[bigram_mod[x]] if token not in stopwords])
essays_df.to_csv("temp.csv")
stop = time.time()
print("({} s)\n".format(stop-start))

# we don't have any use for the unigram lists anymore - just drop them
essays_df.drop(columns=['unigrams'], inplace=True)

# stem the tokens (optional?)
start = time.time()
print("stemming the tokens...")
stemmer = PorterStemmer()
essays_df['tokens'] = essays_df['tokens'].apply(lambda x: [stemmer.stem(token) for token in x])
essays_df.to_csv("temp.csv")
stop = time.time()
print("({} s)\n".format(stop-start))

# using gensim, create the dictionary and the corpus to input into the LDA model
# from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
start = time.time()
print("creating the gensim dictionary and corpus...")
id2word = corpora.Dictionary(essays_df['tokens']) # Term-Document Frequency
# creates a list of (int,int) tuples, where the first is the unique id of the word, and the second is the number of times it appears in the document
corpus = [id2word.doc2bow(text) for text in essays_df['tokens']]
stop = time.time()
print("({} s)\n".format(stop-start))

start = time.time()
print("creating the lda model...")
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, # number of topics to extract
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
stop = time.time()
print("({} s)\n".format(stop-start))

# print(lda_model.print_topics())

start = time.time()
print("creating a dataframe to relate topics to essays...")
sent_topics_df = pd.DataFrame()
for i, row in enumerate(lda_model[corpus]):
    row = sorted(row[0], key=lambda x: x[1], reverse=True)
    (topic_num, prop_topic) = row[0]
    wp = lda_model.show_topic(topic_num)
    topic_keywords = ", ".join([word for word, prop in wp])
    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
sent_topics_df.columns = ['Dominant_Topic', 'Percent_Contribution', 'Topic_Keywords']

print("exporting topics...")
essays_df.reset_index(inplace=True)
essays_df = essays_df.merge(sent_topics_df, left_index=True, right_index=True)
essays_df[['_projectid', 'Dominant_Topic', 'Percent_Contribution', 'Topic_Keywords']].to_csv("lda_essay.csv", index=None)
stop = time.time()
print("({} s)\n".format(stop-start))
