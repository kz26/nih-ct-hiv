#!/usr/bin/env python3

import random
import re
import sqlite3
import sys

import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from re_analyze import score_text as re_score_text


# signatures for line filtering
SIGNATURES = (
    (r'HIV', 0),
    (r'human immunodef', re.IGNORECASE),
    (r'immunodef', re.IGNORECASE),
    (r'immuno-?com', re.IGNORECASE),
    (r'(presence|uncontrolled).+(disease|illness|condition)', re.IGNORECASE),
    (r'immune comp', re.IGNORECASE),
)

REGEXES = [re.compile(x[0], flags=x[1]) for x in SIGNATURES]


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
        raise Exception("No annotation for %s" % id)
    else:
        return result[0]


def filter_study(study_text):
    """take one study and return one or more relevant lines along with its inclusion/exclusion context"""
    chunks = re.split("^(.*(?:inclusion|include|exclusion|exclude).*)$", study_text, flags=re.MULTILINE | re.IGNORECASE)
    lines = []
    for blk in chunks:
        blk = blk.strip()
        if re.search('exclusion|exclude|not [A-Z-a-z]*eligible|ineligible', blk.lower()):
            inclusion = -1
        elif re.search('inclusion|include|eligible', blk.lower()):
            inclusion = 1
        else:
            inclusion = 0
        pre = None
        segments = re.split(r'(\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|[A-Za-z]+ ?: +|!(?:[a-z]{,3} |including )[A-Z][a-z]+ )', blk, flags=re.MULTILINE)
        for i, l in enumerate(segments):
            m_pre = re.match(r'[A-Z][a-z]+ ', l)
            if m_pre:
                pre = l
                if i != len(segments) - 1:
                    continue
            if l:
                if pre:
                    l = pre + l
                    pre = None
                l = l.strip()
                if l:
                    if line_match(l):
                        lines.append((l, inclusion))
    return lines


def vectorize_all(vectorizer, input_lines, fit=False):
    if fit:
        dtm = vectorizer.fit_transform([x[0] for x in input_lines])
    else:
        dtm = vectorizer.transform([x[0] for x in input_lines])
    ie_status_m = coo_matrix([[x[1]] for x in input_lines])
    dtm = normalize(hstack([dtm, ie_status_m]))
    return dtm


if __name__ == '__main__':
    for x in REGEXES:
        print(x)

    COMBINE_RE = False
    if len(sys.argv) > 2 and sys.argv[2].lower() == 'true':
        COMBINE_RE = True

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()
    c.execute('SELECT t1.NCTId, t1.BriefTitle, t1.Condition, t1.EligibilityCriteria, t2.hiv_eligible FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')

    X_training = []
    y_training = []
    X_test = []
    X_test_raw = []
    y_true = []
    y_test_text = []
    y_true_text = []
    test_line_map = []   # line ranges for each study

    train_count = 0
    train_positive = 0
    test_positive = 0
    test_labels = []
    for row in c.fetchall():
        lines = filter_study('\n'.join(row[1:4]))
        if lines:
            if random.random() >= 0.4:
                X_training.extend(lines)
                y_training.extend([row[4]] * len(lines))
                train_count += 1
                if row[4]:
                    train_positive += 1
            else:
                X_test_raw.append(row[3])
                sp = len(X_test)
                X_test.extend(lines)
                test_line_map.append((sp, len(X_test)))
                y_true.extend([row[4]] * len(lines))
                y_true_text.append(row[4])
                if row[4]:
                    test_positive += 1
                test_labels.append(row[0])
        else:
            print("[WARNING] no lines returned from %s" % row[0])

    vectorizer = CountVectorizer(ngram_range=(2, 2), binary=True)
    X_training = vectorize_all(vectorizer, X_training, fit=True)
    X_test = vectorize_all(vectorizer, X_test)

    #model = MultinomialNB()
    #model = LogisticRegression(class_weight='balanced')
    #model = SGDClassifier(loss='log', n_iter=100)
    #model = svm.SVC(probability=True)
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_training, y_training)

    probabilities = model.predict_proba(X_test)
    probabilities_text = []

    for i, r in enumerate(test_line_map):
        ps = [x[1] for x in probabilities[r[0]:r[1]]]
        if COMBINE_RE:
            ps.append(re_score_text(test_labels[i], X_test_raw[i]))
        probabilities_text.append(ps)
        if np.average(ps) >= 0.05 or max(ps) >= 0.1:
            cps = 1
        else:
            cps = 0
        y_test_text.append(cps)

    true_scores = y_true_text
    predicted_scores = y_test_text
    assert(len(true_scores) == len(predicted_scores) == len(test_labels))

    mismatches_fp = []
    mismatches_fn = []
    for i in range(len(true_scores)):
        if true_scores[i] != predicted_scores[i]:
            if predicted_scores[i] == 0:
                mismatches_fn.append([test_labels[i], probabilities_text[i]])
            else:
                mismatches_fp.append([test_labels[i], probabilities_text[i]])
        elif true_scores[i] == 1:
            print(probabilities_text[i])
    print("FP        : %s" % str(mismatches_fp))
    print("FN        : %s" % str(mismatches_fn))
    print("Trn count : %s" % train_count)
    print("Training +: %s" % train_positive)
    print("Test count: %s" % len(true_scores))
    print("Test +    : %s" % test_positive)
    print("Accuracy  : %s" % accuracy_score(true_scores, predicted_scores))
    print(classification_report(true_scores, predicted_scores, target_names=['HIV-ineligible', 'HIV-eligible']))
    print("AUC:      : %s" % roc_auc_score(true_scores, predicted_scores))
    print("Confusion matrix:")
    print(confusion_matrix(true_scores, predicted_scores))
