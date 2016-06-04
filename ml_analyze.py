#!/usr/bin/env python3

import random
import re
import sqlite3
import sys

from manual_annotator import annotate_interactive

import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron


# signatures for line filtering
SIGNATURES = (
    r'HIV',
    r'human immunodef',
    r'immunodef',
    r'immunocom',
    r'immunosup'
)

REGEXES = [re.compile(x) for x in SIGNATURES]


def line_match(line):
    for rx in REGEXES:
        if rx.search(line):
            return True
    return False


def get_true_hiv_status(conn, id):
    c = conn.cursor()
    c.execute("SELECT hiv_eligible FROM hiv_status WHERE NCTId=?", [id])
    result = c.fetchone()
    if result is None:
        return annotate_interactive(conn, id)
    else:
        return result[0]


def filter_study(study_text):
    """take one study and return one or more relevant lines along with its inclusion/exclusion context"""
    #chunks = re.split(".{,15}(inclusion|exclusion).{,15}$", study_text, flags=re.MULTILINE | re.IGNORECASE)
    chunks = re.split("(criteri[A-Z ]*|.{,20})(inclusion|exclusion)([A-Z ]*criteri|.{,20})$", study_text, flags=re.MULTILINE | re.IGNORECASE)
    inclusion = True
    lines = []
    for blk in chunks:
        blk = blk.strip()
        if 'inclusion' in blk.lower():
            inclusion = True
        elif 'exclusion' in blk.lower():
            inclusion = False
        pre = None
        for l in re.split(r'(\n+|\. +|[A-Za-z]+ ?: +|[A-Z][a-z]+ )', blk, flags=re.MULTILINE):
            m_pre = re.match(r'[A-Z][a-z]+ ', l)
            if m_pre:
                pre = l
                continue
            if l:
                if pre:
                    l = pre + l
                    pre = None
                l = l.strip()
                if l:
                    #print(l)
                    if line_match(l):
                        lines.append((l, inclusion))
    return lines


def vectorize_all(vectorizer, input_lines, fit=False):
    if fit:
        dtm = vectorizer.fit_transform([x[0] for x in input_lines])
    else:
        dtm = vectorizer.transform([x[0] for x in input_lines])
    ie_status_m = coo_matrix([[int(x[1])] for x in input_lines])
    dtm = hstack([dtm, ie_status_m])
    return dtm


if __name__ == '__main__':
    print("Line signatures: %s" % len(REGEXES))

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()
    c.execute('SELECT t1.NCTId, t1.EligibilityCriteria, t2.hiv_eligible FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')

    X_training = []
    y_training = []
    X_test = []
    y_true = []
    y_test_text = []
    y_true_text = []
    test_line_map = []   # line ranges for each study

    train_count = 0
    train_positive = 0
    test_positive = 0
    labels = []
    for row in c.fetchall():
        labels.append(row[0])
        lines = filter_study(row[1])
        if random.random() >= 0.4:
            if row[2] or random.random() >= 0.85:  # undersample negatives
                X_training.extend(lines)
                y_training.extend([row[2]] * len(lines))
                train_count += 1
            if row[2]:
                train_positive += 1
        else:
            sp = len(X_test)
            X_test.extend(lines)
            test_line_map.append((sp, len(X_test)))
            y_true.extend([row[2]] * len(lines))
            y_true_text.append(row[2])
            if row[2]:
                test_positive += 1

    vectorizer = CountVectorizer(ngram_range=(2, 3))
    X_training = vectorize_all(vectorizer, X_training, fit=True)
    X_test = vectorize_all(vectorizer, X_test)

    model = MultinomialNB()
    model.fit(X_training, y_training)

    predictions = model.predict(X_test)

    for i in test_line_map:
        ps = predictions[i[0]:i[1]]
        y_test_text.append(int(round(np.average(ps), 0)))

    true_scores = y_true_text
    predicted_scores = y_test_text

    mismatches = []
    for i in range(len(true_scores)):
        if true_scores[i] != predicted_scores[i]:
            mismatches.append(labels[i])
    print("Trn count : %s" % train_count)
    print("Training +: %s" % train_positive)
    print("Test count: %s" % len(true_scores))
    print("Test +    : %s" % test_positive)
    print("Incorrect : %s" % str(mismatches))
    print("Accuracy  : %s" % accuracy_score(true_scores, predicted_scores))
    print("Precision : %s" % precision_score(true_scores, predicted_scores))
    print("Recall    : %s" % recall_score(true_scores, predicted_scores))
    print("F score   : %s" % f1_score(true_scores, predicted_scores))
    print("AUC:      : %s" % roc_auc_score(true_scores, predicted_scores))
    print("Confusion matrix:")
    print(confusion_matrix(true_scores, predicted_scores))
