import pandas as pd
import re
import scipy
import collections
import numpy
import os
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling import LFAnalysis

data_dir = 'data'
data_file = 'train_test_dataset.csv'
file = os.path.join(data_dir, data_file)
data = pd.read_csv(file, usecols=[0, 1])

STAY = 0
LEAVE = 1
ABSTAIN = -1


# Labeling functions for tweets that suggest favoring Britain to "stay" in EU
#TODO: rewrite these labeling functions using keyword-lookup (see snorkel tutorial)
@labeling_function()
def usual_hashtags_stay(df):
    # TODO: programmatically get common hashtags
    hashtags = ["#SayYes2Europe", "#StrongerIN", "#bremain", "#Stay", "#ukineu", "#votein", "#betteroffin",
                "#leadnotleave", "#VoteYES", "#yes2eu", "#yestoeu"]
    text = df.tweets.lower()
    for hashtag in hashtags:
        if hashtag in text:
            return STAY

    return ABSTAIN


@labeling_function()
def usual_texts_stay(df):
    vote_words = ["vote", "voting", "voted", "choose", "chose"]
    stay_words = ["stay", "remain", ""]

    vote_found, stay_found = False, False
    text = df.tweets.lower()
    for word in text.strip().split():
        if word in vote_words:
            vote_found = True

        if word in stay_words:
            stay_found = True

    return STAY if vote_found and stay_found else ABSTAIN


# Labeling functions for tweets that suggest favoring Britain to "leave" EU
@labeling_function()
def usual_hashtags_leave(df):
    # TODO: programmatically get common hashtags
    # HASHTAG_LEAVE = r"(?i)euroscepticism|beLeave|betteroffout|britainout|LeaveEU|noTTIP|TakeControl|VoteLeave|VoteNO|voteout|end-of-europe|leaveeuofficial|NoThanksEU|nothankseu|ukleave-eu|vote-leave|leaving EU|strongOut|voteLeave|brexitnow|leaveEUOfficial"
    hashtags = ["euroscepticism", "beLeave", "betteroffout", "britainout", "LeaveEU", "noTTIP", "TakeControl",
                "VoteLeave", "VoteNO", "voteout", "end-of-europe", "leaveeuofficial", "NoThanksEU", "nothankseu",
                "ukleave-eu", "vote-leave", "leaving EU", "strongOut", "voteLeave", "brexitnow", "leaveEUOfficial"]
    hashtags = ["#" + hashtag for hashtag in hashtags]

    text = df.tweets.lower()
    for hashtag in hashtags:
        if hashtag in text:
            return LEAVE

    return ABSTAIN


@labeling_function()
def usual_texts_leave(df):
    vote_words = ["vote", "voting", "voted", "choose", "chose", "choosing"]
    stay_words = ["leave", "exit", ""]

    vote_found, stay_found = False, False
    text = df.tweets.lower()
    for word in text.strip().split():
        if word in vote_words:
            vote_found = True

        if word in stay_words:
            stay_found = True

    return LEAVE if vote_found and stay_found else ABSTAIN


lfs = [usual_hashtags_stay, usual_texts_stay, usual_texts_leave, usual_hashtags_leave]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(data)
print(LFAnalysis(L_train, lfs=lfs).lf_summary())