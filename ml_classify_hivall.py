#!/usr/bin/env python3

import pickle
import re
import sqlite3
import string

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as ST
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest

DATABASE = 'studies_all.sqlite'
MODEL = 'cancer_hiv_model.pickle'

REMOVE_PUNC = str.maketrans({key: None for key in string.punctuation})


def filter_study(ec):
    """take one study and returns a filtered version with only relevant lines included"""
    lines = []
    segments = re.split(
        r'\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|(?:[A-Z][A-Za-z]+ )+?[A-Z][A-Za-z]+: +|; +| (?=[A-Z][a-z])',
        ec, flags=re.MULTILINE)
    for i, l in enumerate(segments):
        l = l.strip()
        if l:
            l = l.translate(REMOVE_PUNC).strip()
            if l:
                lines.append(l)
    return '\n'.join(lines)


def vectorize_all(vectorizer, input_docs, fit=False):
    if fit:
        dtm = vectorizer.fit_transform(input_docs)
    else:
        dtm = vectorizer.transform(input_docs)
    return dtm


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT t1.NCTId, t1.EligibilityCriteria, t2.annotation_hiv \
               FROM studies AS t1, annotations AS t2 WHERE t1.NCTId=t2.NCTId AND t2.annotation_hiv IS NOT NULL \
               ORDER BY t1.NCTId")

    X = []
    y = []
    study_ids = []

    count = 0
    for row in c.fetchall():
        text = filter_study(row[1])
        if text:
            yv = row[2]
            if yv == 3:
                yv = 2
            X.append(text)
            y.append(yv)
            study_ids.append(row[0])
        else:
            print("[WARNING] no text returned from %s after filtering" % row[0])

    study_ids = np.array(study_ids)

    model_import = pickle.load(open(MODEL, 'rb'))
    vectorizer = model_import['vectorizer']
    chi2_kbest = model_import['chi2_kbest']
    model = model_import['model']

    X = vectorize_all(vectorizer, X)
    X = chi2_kbest.transform(X)
    y = np.array(y)
    label_map = ('HIV-ineligible', 'indeterminate', 'HIV-eligible')
    print(X.shape)

    seed = 0

    y_predicted = model.predict(X)
    y_predicted_score = model.decision_function(X)
    prob_min = y_predicted_score.min()
    prob_max = y_predicted_score.max()
    y_predicted_proba = []
    for x in y_predicted_score:
        p = [(i - prob_min) / (prob_max - prob_min) for i in x]
        y_predicted_proba.append(p)

    stats = list(metrics.precision_recall_fscore_support(y, y_predicted, beta=2, average=None))[:3]
    rocs = []
    praucs = []
    ap_score = []
    fpr = {}
    tpr = {}
    y_bin = {}
    y_predicted_proba_bin = {}
    for i, label in enumerate(label_map):
        bt = (y == i)
        bp = y_predicted_score[:,i]
        y_bin[i] = bt
        y_predicted_proba_bin[i] = bp

        rocs.append(metrics.roc_auc_score(bt, bp))
        fpr[i], tpr[i], thresholds = metrics.roc_curve(bt, bp)
        praucs.append(metrics.average_precision_score(bt, bp))

    stats.append(tuple(rocs))
    stats.append(tuple(praucs))
    print(stats)

    y_predicted_proba = np.array(y_predicted_proba)

    results = []
    for i in range(len(y)):
        results.append((study_ids[i], y_predicted[i], y[i], y_predicted_proba[i]))
    results.sort(key=lambda x: (x[1], x[2]))
    for x in results:
        print("[%s] %s %s %s" % x)

    for i, label in enumerate(label_map):
        for j, metric in enumerate(('precision', 'recall', 'F2 score', 'ROC-AUC score', 'PR-AUC score')):
            print("%s %s: %s" % (label, metric, stats[j][i]))
        print("%s count: %s" % (label, len([l for l in y_predicted if l == i])))

        plt.figure(1)
        plt.plot(fpr[i], tpr[i],
                 label="%s (mean AUC = %0.2f)" % (label, stats[3][i]), lw=2)
        plt.figure(2)
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_bin[i], y_predicted_proba_bin[i]
        )
        plt.plot(recall, precision,
                 label="%s (PR-AUC = %0.2f)" % (label, stats[4][i]), lw=2)

    print("Confusion matrix:")
    print(metrics.confusion_matrix(y, y_predicted))

    plt.figure(1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    ax = plt.gca()
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    plt.plot(limits, limits, 'k-', alpha=0.75, zorder=0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ML (HIV-all)')
    plt.legend(loc="lower right")

    plt.figure(2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ML (HIV-all)')
    plt.legend(loc="lower left")

    plt.show()

