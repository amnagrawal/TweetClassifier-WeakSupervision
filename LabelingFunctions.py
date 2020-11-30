import pandas as pd
import re
import scipy
import collections
import numpy as np
import os
from sklearn.model_selection import train_test_split
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling import LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds
from preprocess_tweets import preprocess_tweets

STAY = 0
LEAVE = 1
ABSTAIN = -1

# Labeling functions for tweets that suggest favoring Britain to "stay" in EU
#TODO: rewrite these labeling functions using keyword-lookup (see snorkel tutorial)
@labeling_function()
def usual_hashtags_stay(df):
    # TODO: programmatically get common hashtags
    hashtags = ["#SayYes2Europe", "#StrongerIN", "#bremain", "#Stay", "#ukineu", "#votein", "#betteroffin",
                "#leadnotleave", "#VoteYES", "#yes2eu", "#yestoeu", "#Remain"]
    text = df.tweets.lower()
    for hashtag in hashtags:
        if hashtag.lower() in text:
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
    hashtags = ["euroscepticism", "beLeave", "betteroffout", "britainout", "LeaveEU", "noTTIP", "TakeControl",
                "VoteLeave", "VoteNO", "voteout", "end-of-europe", "leaveeuofficial", "NoThanksEU", "nothankseu",
                "ukleave-eu", "vote-leave", "leaving EU", "strongOut", "voteLeave", "brexitnow", "leaveEUOfficial"]
    hashtags = ["#" + hashtag for hashtag in hashtags]

    text = df.tweets.lower()
    for hashtag in hashtags:
        if hashtag.lower() in text:
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


def main():
    pass


if __name__ == "__main__":
    data_dir = 'data'
    data_file = 'train_test_dataset.csv'
    file = os.path.join(data_dir, data_file)
    data = pd.read_csv(file, usecols=[0, 1])

    train_df, test_df = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

    lfs = [usual_hashtags_stay, usual_texts_stay, usual_texts_leave, usual_hashtags_leave]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(train_df)
    print(LFAnalysis(L_train, lfs=lfs).lf_summary())
    # TODO: try to improve coverage

    save_file = os.path.join(data_dir, 'ground_truth_matrix')
    np.save(save_file, L_train, allow_pickle=True)

    golden_labels = []
    for y in train_df.label:
        if y == 'leave':
            golden_labels.append(LEAVE)
        if y == 'stay':
            golden_labels.append(STAY)

    golden_labels = np.asarray(golden_labels)
    save_file = os.path.join(data_dir, 'ground_truth')
    np.save(save_file, golden_labels, allow_pickle=True)

    unlabeled_data = pd.read_csv(os.path.join(data_dir, 'brexit_unlabelled.csv'))
    L_unlabeled = applier.apply(unlabeled_data)
    print(LFAnalysis(L_unlabeled, lfs=lfs).lf_summary())

    save_file = os.path.join(data_dir, 'matrix_for_new_labels')
    np.save(save_file, L_unlabeled, allow_pickle=True)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    L_test = applier.apply(test_df)
    # to_numerical = lambda x: x=='leave'
    # Y_test = [to_numerical(item) for item in test_df.label]
    Y_test = []
    for item in test_df.label:
        if item == 'stay':
            Y_test.append(STAY)
        else:
            Y_test.append(LEAVE)

    Y_test = np.asarray(Y_test)
    label_model_performance = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random",
                                                metrics=['accuracy', 'precision', 'recall', 'f1'])
    print(f"Label Model Accuracy: {label_model_performance['accuracy'] * 100:.1f}%")
    predict_probs = label_model.predict_proba(L_unlabeled)
    preds = probs_to_preds(predict_probs)
    pred_labels = []
    for i in range(len(preds)):
        if preds[i]:
            pred_labels.append('leave')
        else:
            pred_labels.append('stay')
    unlabeled_data['label'] = pred_labels
    unlabeled_data.to_csv(os.path.join(data_dir, 'snorkel_labeled_data.csv'), sep=',', index=False)


