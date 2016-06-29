#!/usr/bin/env python3

import re
import sqlite3
import sys

import numpy as np
from sklearn import metrics

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
        segments = re.split(r'(\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|[A-Za-z]+ ?: +|; +)', blk, flags=re.MULTILINE)
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
    for x in REGEXES:
        print(x)

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    true_scores = []
    predicted_scores = []
    labels = []

    c.execute("SELECT t1.NCTId, t1.BriefTitle, t1.Condition, t1.EligibilityCriteria, t2.hiv_eligible \
        FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId AND t1.StudyType LIKE '%Interventional%' \
        ORDER BY t1.NCTId")

    counter = 1
    for row in c.fetchall():
        print(row[0])
        true_scores.append(int(row[4]))
        predicted_scores.append(score_text(row[0], '\n'.join(row[1:4])))
        labels.append(row[0])
        counter += 1

    true_scores = np.array(true_scores)
    predicted_scores = np.array(predicted_scores)

    print("Count    : %s" % len(true_scores))
    print("Accuracy : %s" % metrics.accuracy_score(true_scores, predicted_scores))
    f2s = []
    prauc = []
    for i in range(3):
        bt = (true_scores == i)
        bp = (predicted_scores == i)
        f2s.append(metrics.fbeta_score(bt, bp, beta=2.0))
        prauc.append(metrics.average_precision_score(bt, bp))
    print("F2 score : %s" % f2s)
    print("PR-AUC   : %s" % prauc)
    print(metrics.classification_report(true_scores, predicted_scores,
                                        target_names=['HIV-ineligible', 'indeterminate', 'HIV-eligible']))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(true_scores, predicted_scores))
