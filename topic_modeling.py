import os, re, string, sys, time, argparse

import pandas as pd
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# replace standard print with a version that can update in the console as we go
def print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()

# perform topic modelling on the essay dataset.
# Prints timer messages to console to update it's progress
# params:
#   output: The csv file to output the final results to
#   sample_size: set to anything else to downsample the number of essays (mostly for debugging)
#   num_topics: the number of topics for LDA to separate esasys into
#   vocab_size: the number of terms LDA should use to determine topics
def do_modeling(output="results.csv", sample_size=0, num_topics=12, vocab_size=10000):
    start = time.time()
    print("loading essays...")
    essays_df = pd.read_csv('data/opendata_essays000.gz', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])

    # these are the only two columsn we need at the moment
    essays_df = essays_df[["_projectid", "essay"]]

    stop = time.time()
    print("({} s)\n".format(stop-start))

    # downsample to start with if requested
    if sample_size != 0:
        essays_df = essays_df.sample(sample_size)

    # drop any rows where the essay field is blank
    essays_df = essays_df[essays_df.essay.notnull()]

    # regex to strip all punctuation and replace it with whitespace, then convert to lowercase and split on all whitespace
    start = time.time()
    print("creating unigrams...")
    essays_df['essay'] = essays_df['essay'].apply(lambda x: ' '.join(re.sub(r'[^\w\s]|\r\n',' ',x).lower().split()))
    stop = time.time()
    print("({} s)\n".format(stop-start))

    # build bigram and trigram models with gensim
    # set thresholds higher if we get too many
    # from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    start = time.time()
    print("building bigram/trigram models...")
    bigram = gensim.models.Phrases([x.split() for x in essays_df['essay']], min_count=3000)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    bigram_mod.save("bigrams.all")
    stop = time.time()
    for i in range(50):
        print([x for x in bigram_mod[essays_df.iloc[i].essay.split()] if "_" in x])
    print("({} s)\n".format(stop-start))

    # use the trigram and bigram models from above to get a list of tokens
    # exclude any tokens that appear in the list "stopwords"
    start = time.time()
    print("creating tokens from unigram/bigram/trigrams...")
    stemmer = PorterStemmer()
    essays_df['tokens'] = essays_df['essay'].apply(lambda x: [stemmer.stem(token) for token in bigram_mod[x.split()]])
    print("({} s)\n".format(time.time()-start))

    # using gensim, create the dictionary and the corpus to input into the LDA model
    # from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    start = time.time()
    print("creating the gensim dictionary and corpus...")
    id2word = corpora.Dictionary(essays_df['tokens']) # Term-Document Frequency

    # drop words that don't appear at least 15 times, or appear in more than 1/2 of the documents
    # keep the number of terms specified in vocab_size
    id2word.filter_extremes(no_below=200, no_above=0.25, keep_n=vocab_size)
    # creates a list of (int,int) tuples, where the first is the unique id of the word, and the second is the number of times it appears in the document
    corpus = [id2word.doc2bow(text) for text in essays_df['tokens']]
    stop = time.time()
    print("({} s)\n".format(stop-start))

    start = time.time()
    print("creating the lda model...")
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                       id2word=id2word,
                                                       num_topics=num_topics)
    lda_model.save("lda.model")
    stop = time.time()
    print("({} s)\n".format(stop-start))

    start = time.time()
    print("creating visualization with pyLDAvis...")
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, "vis_out.html")
    print("({} s)\n".format(time.time()-start))

    start = time.time()
    print("saving topic term lists...")
    topic_terms = {}
    for topic in range(num_topics):
        topic_terms[topic] = [id2word[word[0]] for word in lda_model.get_topic_terms(topic, topn=20)]
    with open("topic.terms", 'w') as f:
        f.write(''.join(["{}: {}\n".format(topic,','.join(terms)) for topic,terms in topic_terms.items()]))
    print("({} s)\n".format(time.time()-start))

    start = time.time()
    print("getting top three topics per doc...")
    topic_by_doc = {}
    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        topic_by_doc[i] = [x[0] for x in row][:3]

    # https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    doc_topics_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in topic_by_doc.items()])).transpose()
    doc_topics_df.columns = ["Topic_"+str(x+1) for x in range(3)]
    print("({} s)\n".format(time.time()-start))

    start = time.time()
    print("merging with original dataframe and exporting...")
    essays_df.reset_index(inplace=True)
    essays_df = essays_df.merge(doc_topics_df, left_index=True, right_index=True)
    essays_df.set_index("_projectid", inplace=True)
    essays_df[["Topic_1", "Topic_2", "Topic_3"]].to_csv(output)
    print("({} s)\n".format(time.time()-start))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sample_size", help="How many samples to draw from the data set",
                           type=int, default="0", required=False)
    argparser.add_argument("--output", help="The csv file to write the results to. (default: results.csv)",
                           type=str, default="results.csv", required=False)
    argparser.add_argument("--vocab_size", help="Size of vocabulary",
                           type=int, default=10000, required=False)
    argparser.add_argument("--num_topics", help="Number of topics",
                           type=int, default=12, required=False)
    args = argparser.parse_args()

    overall_start = time.time()
    do_modeling(
        output=args.output,
        sample_size=args.sample_size,
        num_topics=args.num_topics,
        vocab_size=args.vocab_size)
    print("Overall Time: ({} s)\n".format(time.time()-overall_start))
