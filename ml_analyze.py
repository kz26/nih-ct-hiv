#!/usr/bin/env python3

import random
import re
import sqlite3
import string
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn import cross_validation
from scipy import stats as ST


# signatures for line filtering
# SIGNATURES = (
#     (r'HIV', 0),
#     (r'human immunodef', re.IGNORECASE),
#     (r'immunodef', re.IGNORECASE),
#     (r'immuno-?com', re.IGNORECASE),
#     (r'(presence|uncontrolled|severe|chronic).+(disease|illness|condition)', re.IGNORECASE),
#     (r'immune comp', re.IGNORECASE),
#     (r'criteri', re.IGNORECASE),
#     (r'characteristics', re.IGNORECASE),
#     (r'inclusion|include', re.IGNORECASE),
#     (r'exclusion|exclude', re.IGNORECASE)
# )

# REGEXES = [re.compile(x[0], flags=x[1]) for x in SIGNATURES]

REMOVE_PUNC = str.maketrans({key: None for key in string.punctuation})


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
    """take one study and returns a filtered version with only relevant lines included"""
    # return study_text.translate(REMOVE_PUNC)
    lines = []
    pre = None
    segments = re.split(
        r'(\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|(?:[A-Z][A-Za-z]+ )+?[A-Z][A-Za-z]+: +|; +|(?<!\()(?:[A-Z][a-z]+ ))',
        study_text, flags=re.MULTILINE)
    for i, l in enumerate(segments):
        m_pre = re.match(r'[A-Z][a-z]+ ', l)
        if m_pre:
            if i != len(segments) - 1:
                pre = l
                continue
            else:
                pre = None
        if l:
            if pre:
                l = pre + l
                pre = None
            l = l.translate(REMOVE_PUNC).strip()
            if l:
                lines.append(l)
    # print('\n'.join(lines))
    return '\n'.join(lines)


def vectorize_all(vectorizer, input_docs, fit=False):
    if fit:
        dtm = vectorizer.fit_transform(input_docs)
    else:
        dtm = vectorizer.transform(input_docs)
    print(dtm.shape)
    return dtm


if __name__ == '__main__':
    # for x in REGEXES:
    #     print(x)

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()
    c.execute('SELECT t1.NCTId, t1.BriefTitle, t1.Condition, t1.EligibilityCriteria, t2.hiv_eligible FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')

    X = []
    y = []
    study_ids = []

    count = 0
    count_positive = 0
    for row in c.fetchall():
        # print(row[0])
        text = filter_study('\n'.join(row[1:4]))
        if text:
            X.append(text)
            y.append(row[4])
            study_ids.append(row[0])
            if row[4]:
                count_positive += 1
        else:
            print("[WARNING] no text returned from %s after filtering" % row[0])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorize_all(vectorizer, X, fit=True)
    y = np.array(y)

    chi2_best = SelectKBest(chi2, k=250)
    X = chi2_best.fit_transform(X, y)
    print(X.shape)

    stats = []
    seed = 0
    skf = cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True, random_state=seed)
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        #model = MultinomialNB()
        #model = LogisticRegression(class_weight='balanced')
        #model = SGDClassifier(loss='hinge', n_iter=100, penalty='elasticnet')
        #model = svm.SVC(C=1000000, kernel='linear', class_weight={1: 10, 2: 10})
        model = svm.LinearSVC(C=150, class_weight={1: 5, 2: 12}, random_state=seed)
        #model = RandomForestClassifier(class_weight='balanced')
        #model = AdaBoostClassifier(n_estimators=100)

        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)
        sd = list(metrics.precision_recall_fscore_support(y_test, y_predicted, average=None))[:3]
        aucs = []
        for i in range(3):
            bt = (y_test == i)
            bp = (y_predicted == i)
            aucs.append(metrics.roc_auc_score(bt, bp))
            # fpr, tpr, thresholds = metrics.roc_curve(bt, bp)
            # aucs.append(metrics.auc(fpr, tpr))
        sd.append(tuple(aucs))
        stats.append(sd)
        # target_names = ['HIV-ineligible', 'indeterminate', 'HIV-eligible']
        # print(classification_report(y_test, y_predicted, target_names=target_names))

    for i, label in enumerate(('HIV-ineligible', 'indeterminate', 'HIV-eligible')):
        for j, metric in enumerate(('precision', 'recall', 'F1 score', 'ROC-AUC score')):
            sd = [x[j][i] for x in stats]
            sd_mean = np.mean(sd)
            sd_ci = ST.t.interval(0.95, len(sd) - 1, loc=sd_mean, scale=ST.sem(sd))
            print("%s %s: %s %s" % (label, metric, sd_mean, sd_ci))

    # print("Count     : %s" % len(true_scores))
    # print("CV folds  : %s" % cross_validation_count)
    # print("Accuracy  : %s" % accuracy_score(true_scores, predicted_scores))
    # # print("ROC-AUC   : %s" % roc_auc_score(true_scores, predicted_scores))
    # 
    
    # print("Confusion matrix:")
    # print(confusion_matrix(true_scores, predicted_scores))
