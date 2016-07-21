#!/usr/bin/env python3

import re
import sqlite3

import numpy as np
from sklearn import metrics, cross_validation
from scipy import stats as ST

DATABASE = 'studies.sqlite'

ALWAYS_POSITIVE_SIGNATURES = (
    r'(HIV|human immunodeficiency virus) testing is not required',
    r'asymptomatic.+(HIV|human immunodeficiency virus)',
)

POSITIVE_ONLY_SIGNATURES = (
    r'patients (with|having).+(HIV|human immunodeficiency virus)',
    r'anti.+(HIV|human immunodeficiency virus).+antibody',
)
POSITIVE_SIGNATURES = (
    r'seropositive for (HIV|human immunodeficiency virus)',
    r'positiv.*?(HIV|human immunodeficiency virus).+?antibody',
    r'any form of primary|secondary immunodeficiency',
    r'history of.+(HIV|human immunodeficiency virus)',
    r'History of.+(?:primary|secondary) immunodeficiency',
    r'(HIV|human immunodeficiency virus).+antibody[A-Z ]+positive',
    r'known diagnosis of.+?HIV/AIDS',
    r'test positive for.+?(HIV|human immunodeficiency virus)',
    r'HIV \(Human Immunodeficiency Virus\) positive'
    r'(documentation|evidence) of.+?(HIV|human immunodeficiency virus)',
    r'(HIV|human immunodeficiency virus).+?(HAART|retroviral).+?',
    r'(known )?human immunodeficiency virus \(HIV\) infection',
    r'(known )?infection with (HIV|human immunodeficiency virus)',
    r'known[A-Z0-9 -,/]+?(HIV|human immunodeficiency virus)',
    r'diagnosis of (HIV|human immunodeficiency virus) infection',
    r'(HIV|human immunodeficiency virus).+?infections?',
    r'infect[A-Z0-9 -,/]+?(HIV|human immunodeficiency virus)',
    r'positiv[A-Z0-9 -,/]+?(HIV|human immunodeficiency virus)',
    r'(?:active|chronic)[A-Z0-9 -,/]+?(HIV|human immunodeficiency virus)',
    r'(?:active|chronic)[A-Z0-9 -,/]+?(infection)',
    r'(HIV|human immunodeficiency virus)(-| )positiv',
    r'risk of[A-Z0-9 -,/]+?(HIV|human immunodeficiency virus)',
    r'immunodeficiency[A-Z0-9 -,/]+(HIV|human immunodeficiency virus)',
    r'patients? (who have|with).+(HIV|human immunodeficiency virus)',
    r'clinically (evident|significant)[A-Z0-9 -,/]+?(HIV|human immunodeficiency virus)',
    r'suffering from[A-Z0-9 -,/]+?(HIV|human immunodeficiency virus)',
    r'HIV-seropositive',
    r'HIV infection',
    r'HIV\+',
)

NEGATIVE_SIGNATURES = (
    r'HIV-( +|$)',
    r'no reactive HIV testing',
)

ALWAYS_POSITIVE_REGEXES = [(re.compile(x, flags=re.IGNORECASE), True) for x in ALWAYS_POSITIVE_SIGNATURES]
POSITIVE_ONLY_REGEXES = [(re.compile(x, flags=re.IGNORECASE), 1) for x in POSITIVE_ONLY_SIGNATURES]
POSITIVE_REGEXES = [(re.compile(x, flags=re.IGNORECASE), 1) for x in POSITIVE_SIGNATURES]
NEGATIVE_REGEXES = [(re.compile(x, flags=re.IGNORECASE), -1) for x in NEGATIVE_SIGNATURES]

POSITIVE_SUFFIX_REGEXES = [re.compile(x, flags=re.IGNORECASE) for x in (
    r'(HIV|human immunodeficiency virus)[A-Z0-9 -,\(\)]+(may be|are|possibl(e|y)) (eligible|permitted)',
    r'(HIV|human immunodeficiency virus)[A-Z0-9 -,\(\)]+not[A-Z0-9 -,]+excluded',
    r'(HIV|human immunodeficiency virus)[A-Z0-9 -,\(\)]+discretion'
)]

for x in POSITIVE_SIGNATURES:
    if 'positiv' in x:
        NEGATIVE_REGEXES.append((re.compile(x.replace('positiv', 'negativ'), flags=re.IGNORECASE), -1))
    NEGATIVE_REGEXES.append((re.compile(r'not? [A-Z0-9 -,]*?' + x, flags=re.IGNORECASE), -1))

REGEXES = ALWAYS_POSITIVE_REGEXES + NEGATIVE_REGEXES + POSITIVE_ONLY_REGEXES + POSITIVE_REGEXES


def score_text(label, text):
    if not (re.search('HIV', text) or re.search('human immunodeficiency virus', text, flags=re.IGNORECASE)):
        return 1
    chunks = re.split(r"^(.*(?:criteri|characteristics).*)$", text, flags=re.MULTILINE | re.IGNORECASE)
    score = 0
    multiplier = 1
    for blk in chunks:
        blk = blk.strip()
        if re.search(r'criteri|characteristics', blk, flags=re.IGNORECASE):
            if re.search('exclusion|exclude|non.?inclusion|not [A-Z-a-z]*eligible|ineligible', blk.lower()):
                multiplier = -1
                print("[EXCLUSION BLOCK]")
            elif re.search('inclusion|include|eligible', blk.lower()):
                multiplier = 1
                print("[INCLUSION BLOCK]")
        pre = None
        segments = re.split(r'(\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|[A-Za-z]+ ?: +|; +)', blk,
                            flags=re.MULTILINE)
        for i, l in enumerate(segments):
            if l:
                l = l.strip()
                matched = False
                for rx, v in REGEXES:
                    if rx.search(l):
                        matched = True
                        s = v * multiplier  # the default
                        # handle special cases
                        if v is True or (multiplier == 1 and v == 1):
                            s = 2
                        elif v == 1:
                            for sx in POSITIVE_SUFFIX_REGEXES:
                                if sx.search(l):
                                    s = 1
                                    break
                        score += s
                        print("[%s, %s] (%s): %s" % (label, s, rx, l))
                        break
                if not matched:
                    print("[%s, UNKNOWN] %s" % (label, l))
    print("[%s] Score: %s" % (label, score))
    if score > 0:
        return 2
    elif score < 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    np.set_printoptions(precision=2)

    for x in REGEXES:
        print(x)

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    X = []
    y = []
    study_ids = []

    c.execute("SELECT t1.NCTId, t1.BriefTitle, t1.Condition, t1.EligibilityCriteria, t2.hiv \
        FROM studies AS t1, annotations AS t2 WHERE t1.NCTId=t2.NCTId AND t1.StudyType LIKE '%Interventional%' \
        AND t2.hiv IS NOT NULL ORDER BY t1.NCTId")

    counter = 0
    for row in c.fetchall():
        study_ids.append(row[0])
        X.append('\n'.join(row[1:4]))
        yv = row[4]
        if yv == 3:
            yv = 2
        y.append(yv)
        counter += 1

    X = np.array(X)
    y = np.array(y)

    study_ids = np.array(study_ids)
    label_map = ('HIV-ineligible', 'indeterminate', 'HIV-eligible')

    seed = 0
    folds = 10
    skf = cross_validation.StratifiedKFold(y, n_folds=folds, shuffle=True, random_state=seed)
    y_test_all = []
    y_pred_all = []
    stats = []
    global_stats = []
    for train, test in skf:
        X_test, y_test = X[test], y[test]
        y_test_all.extend(y_test)
        y_pred = []
        for sid, text in zip(study_ids[test], X_test):
            y_pred.append(score_text(sid, text))
        y_pred = np.array(y_pred)
        y_pred_all.extend(y_pred)

        sd = list(metrics.precision_recall_fscore_support(y_test, y_pred, beta=2, average=None))[:3]
        aucs = []
        ap_score = []
        for i, label in enumerate(label_map):
            bt = (y_test == i)
            bp = (y_pred == i)

            aucs.append(metrics.roc_auc_score(bt, bp))
            ap_score.append(metrics.average_precision_score(bt, bp))

        sd.append(np.array(aucs))
        sd.append(np.array(ap_score))
        stats.append(sd)

        global_sd = list(metrics.precision_recall_fscore_support(
            y_test, y_pred, beta=2.0, average='macro'))[:3]
        global_stats.append(global_sd)

    for i, label in enumerate(label_map):
        stat_mean = {}
        for j, metric in enumerate(('precision', 'recall', 'F2 score', 'ROC-AUC score', 'PR-AUC score')):
            sd = np.array([x[j][i] for x in stats])
            print("%s %s: %s" % (label, metric, sd))
            sd_mean = np.mean(sd)
            stat_mean[metric] = sd_mean
            sd_ci = np.array(ST.t.interval(0.95, len(sd) - 1, loc=sd_mean, scale=ST.sem(sd)))
            print("%s %s: %.2f %s" % (label, metric, sd_mean, sd_ci))
        print("%s count: %s" % (label, len([x for x in y_test_all if x == i])))

    stat_mean = {}
    for i, metric in enumerate(('precision', 'recall', 'F2 score')):
        sd = np.array([x[i] for x in global_stats])
        print("All %s: %s" % (metric, sd))
        sd_mean = np.mean(sd)
        stat_mean[metric] = sd_mean
        sd_ci = np.array(ST.t.interval(0.95, len(sd) - 1, loc=sd_mean, scale=ST.sem(sd)))
        print("All %s: %.2f %s" % (metric, sd_mean, sd_ci))

    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test_all, y_pred_all))
