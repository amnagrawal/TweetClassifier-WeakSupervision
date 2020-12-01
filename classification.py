from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


def train_model(classifier, trainx, trainy, testx, testy, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(trainx, trainy)

    # predict the labels on validation dataset
    predictions = classifier.predict(testx)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return accuracy_score(predictions, testy), confusion_matrix(predictions, testy)


def classify(data, method):
    train_x, test_x, train_y, test_y = train_test_split(data.preprocess_tweets, data.label, stratify=data.label)
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.transform(test_y)
    # using tf-idf at word level
    tfidf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vector.fit(data.tweets)
    xtrain_tfidf = tfidf_vector.transform(train_x)
    xtest_tfidf = tfidf_vector.transform(test_x)

    accuracy, cm = train_model(LogisticRegression(), xtrain_tfidf, train_y, xtest_tfidf, test_y)
    print(f"Logistic Regression accuracy {method}: {str(round(accuracy, 3))}")
    df_cm = np.array(cm)
    sn.heatmap(cm, annot=True, annot_kws={"size": 16}, xticklabels=['stay', 'leave'], yticklabels=['stay', 'leave'],
               fmt="d")
    plt.title(method)
    # plt.xlabel("True values ")
    plt.ylabel("Predicted values")
    plt.show()


if __name__ == '__main__':
    data_dir = 'data'
    data_file = 'training_dataset.csv'
    data = pd.read_csv(os.path.join(data_dir, data_file))
    classify(data, "without weak supervision")

    snorkel_data_file = 'new_labeled_data.csv'
    data2 = pd.read_csv(os.path.join(data_dir, snorkel_data_file))
    classify(data2, "with weak supervision")