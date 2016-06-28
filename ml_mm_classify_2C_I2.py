#!/usr/bin/env python3
# Binary classifier - combine classes 1 (indeterminate) and 2 (HIV-eligible)

import pickle
import re
import sqlite3
import string
import sys


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import svm

from sklearn.feature_selection import chi2, SelectKBest
from sklearn import cross_validation
from scipy import stats as ST
import matplotlib.pyplot as plt

DATABASE = 'studies.sqlite'
CUI_PATH = 'cuis_I.pickle'
STUDY_FILE_PATH = 'mentions_hiv.txt'

REMOVE_PUNC = str.maketrans({key: None for key in string.punctuation})


def filter_study(title, condition, ec):
    """take one study and returns a filtered version with only relevant lines included"""
    lines = [title, condition]
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
    X = []
    y = []
    study_ids = []

    sf = open(STUDY_FILE_PATH, 'r')
    sf_sids = [l.strip() for l in sf if l.strip()]

    CUI = pickle.load(open(CUI_PATH, 'rb'))

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    for study_id in sf_sids:
        c.execute("SELECT t1.NCTId, t1.BriefTitle, t1.Condition, t1.EligibilityCriteria, t2.hiv_eligible \
            FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId AND t2.NCTId=?", [study_id])
        row = c.fetchone()
        text = filter_study(row[1], row[2], row[3]) + '\n' + '\n'.join(CUI[row[0]])
        if text:
            X.append(text)
            yv = row[4]
            if yv == 2:
                yv = 1  # remap 0-1-2 to 0-1
            y.append(yv)
            study_ids.append(row[0])
        else:
            sys.stderr.write("[WARNING] no text returned from %s after filtering\n" % row[0])

    study_ids = np.array(study_ids)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorize_all(vectorizer, X, fit=True)
    y = np.array(y)

    chi2_best = SelectKBest(chi2, k=500)
    X = chi2_best.fit_transform(X, y)

    stats = []
    seed = 0
    folds = 10
    sys.stderr.write("CV folds: %s\n" % folds)

    label_map = ('HIV-ineligible', 'HIV-eligible')
    mean_fpr = {}
    mean_tpr = {}
    y_test_class = {}
    y_pred_class = {}
    for x in label_map:
        mean_fpr[x] = np.linspace(0, 1, 100)
        mean_tpr[x] = [0.0]
        y_test_class[x] = []
        y_pred_class[x] = []

    study_ids_test = []
    y_test_all = []
    y_pred_all = []
    y_pred_proba_all = []

    skf = cross_validation.StratifiedKFold(y, n_folds=folds, shuffle=True, random_state=seed)
    counter = 0
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        y_test_all.extend(y_test)
        study_ids_test.extend(list(study_ids[test]))

        model = svm.LinearSVC(C=100, class_weight={0: 8}, random_state=seed)
        model.fit(X_train, y_train)

        y_predicted = model.predict(X_test)
        y_pred_all.extend(y_predicted)

        y_predicted_score = model.decision_function(X_test)
        prob_min = y_predicted_score.min()
        prob_max = y_predicted_score.max()
        for x in y_predicted_score:
            p = (x - prob_min) / (prob_max - prob_min)
            y_pred_proba_all.append((1-p, p))

        sd = list(metrics.precision_recall_fscore_support(y_test, y_predicted, beta=2, average=None))[:3]
        aucs = []
        ap_score = []
        for i, label in enumerate(label_map):
            bt = (y_test == i)
            if i == 0:
                bp = [1-x for x in y_predicted_score]
            else:
                bp = [x for x in y_predicted_score]
            y_test_class[label].extend(list(bt))
            y_pred_class[label].extend(bp)

            aucs.append(metrics.roc_auc_score(bt, bp))
            fpr, tpr, thresholds = metrics.roc_curve(bt, bp)
            mean_tpr[label] += np.interp(mean_fpr[label], fpr, tpr)
            mean_tpr[label][0] = 0.0

            ap_score.append(metrics.average_precision_score(bt, bp))

        sd.append(tuple(aucs))
        sd.append(tuple(ap_score))
        stats.append(sd)

        counter += 1

    y_pred_proba_all = np.array(y_pred_proba_all)

    results = []
    for i in range(len(y_test_all)):
        results.append((study_ids_test[i], y_pred_all[i], y_test_all[i], y_pred_proba_all[i]))
    results.sort(key=lambda x: (x[1], x[2]))
    for x in results:
        if x[1] == 1:
            print(x[0])

    for i, label in enumerate(label_map):
        stat_mean = {}
        for j, metric in enumerate(('precision', 'recall', 'F2 score', 'ROC-AUC score', 'PR-AUC score')):
            sd = [x[j][i] for x in stats]
            sd_mean = np.mean(sd)
            stat_mean[metric] = sd_mean
            sd_ci = ST.t.interval(0.95, len(sd) - 1, loc=sd_mean, scale=ST.sem(sd))
            sys.stderr.write("%s %s: %s %s\n" % (label, metric, sd_mean, sd_ci))

        plt.figure(1)
        mean_tpr[label] /= folds
        mean_tpr[label][-1] = 1.0
        plt.plot(mean_fpr[label], mean_tpr[label],
                 label="%s (mean AUC = %0.2f)" % (label, stat_mean['ROC-AUC score']), lw=2)
        plt.figure(2)
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_class[label], y_pred_class[label]
        )
        plt.plot(recall, precision,
                 label="%s (PR-AUC = %0.2f)" % (label, stat_mean['PR-AUC score']), lw=2)

    sys.stderr.write("Confusion matrix:\n")
    sys.stderr.write(str(metrics.confusion_matrix(y_test_all, y_pred_all)))
    sys.stderr.write('\n')

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
    plt.title('Mean ROC')
    plt.legend(loc="lower right")

    plt.figure(2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")

    plt.show()
