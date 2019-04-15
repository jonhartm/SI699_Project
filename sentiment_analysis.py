import os, re, string, sys, time, argparse
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# If true, creates backup files every 10k essays - these can be deleted once complete.
MAKE_BACKUP_FILES = False

# Does a sentiment analysis of the entire essay and the first sentence of that essay
def do_sentiment_analysis():
    essays_df = pd.read_csv('data/opendata_essays000.gz', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])
    analyzer = SentimentIntensityAnalyzer()

    results = []
    count = 0
    for i,row in essays_df.iterrows():
        try:
            entire_essay = {"entire_essay_{}".format(t):v for t,v in analyzer.polarity_scores(row.essay).items()}
            first_sentence = {"first_sentence_{}".format(t):v for t,v in analyzer.polarity_scores(sent_tokenize(row.essay)[0]).items()}
            # sneak the index in there
            entire_essay['i'] = i

            results.append({**entire_essay, **first_sentence})
        except:
            # We will fail on occasion - There are some blank essays
            print("failed on ",i)

        count += 1
        if count % 1000 == 0:
            print(count)
        if count % 10000 == 0 and MAKE_BACKUP_FILES:
            pd.DataFrame(results).to_csv("sentiment_results_{}.csv".format(str(count).zfill(4)))
            results = []

    # store the results in a csv once we're through
    pd.DataFrame(results).to_csv("sentiment_results.csv")
