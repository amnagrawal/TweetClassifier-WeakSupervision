import numpy as np
import pandas as pd
import csv
import os
import re

from sklearn.utils import shuffle
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(reduce_len=True, preserve_case=False,
        strip_handles=False)

FLAGS = re.MULTILINE | re.DOTALL


def preprocess(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"[!,.?£$%&|\(\)]", '')
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"#\S+", "<hashtag>")

    tokens = tknzr.tokenize(text.lower())
    return ' '.join(tokens)


def preprocess_tweets(df):
    processed_tweets = []
    for tweet in df.tweets:
        processed_tweet = preprocess(tweet)
        processed_tweets.append(processed_tweet)
    df['preprocess_tweets'] = processed_tweets
    return df


if __name__ == "__main__":
    data_dir = "data"
    # labeled_data = np.load(os.path.join(data_dir, 'labeled_tweets.npy'))
    # training_dataset = pd.DataFrame.from_records(labeled_data)
    training_dataset = pd.read_csv(os.path.join(data_dir, 'labeled_tweets.csv'))
    training_dataset = preprocess_tweets(training_dataset)
    training_dataset.to_csv(os.path.join(data_dir, 'training_dataset.csv'), sep=',', index=False)

    snorkel_labeled_data = pd.read_csv(os.path.join(data_dir, 'snorkel_labeled_data.csv'))
    snorkel_labeled_data = preprocess_tweets(snorkel_labeled_data)

    new_labeled_data = pd.concat([snorkel_labeled_data, training_dataset], sort=False, ignore_index=True)
    new_labeled_data.to_csv(os.path.join(data_dir, 'new_labeled_data.csv'), index=False)