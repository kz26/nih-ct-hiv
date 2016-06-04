#!/usr/bin/env python3

import re
import sqlite3
import sys

from manual_annotator import annotate_interactive

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

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
    r'history of (HIV|human immunodeficiency virus)',
    r'History of primary|secondary immunodeficiency',
    r'(HIV|human immunodeficiency virus).+antibody positive',
    r'known diagnosis of.+?HIV/AIDS',
    r'test positive for.+?(HIV|human immunodeficiency virus)',
    r'HIV \(Human Immunodeficiency Virus\) positive'
    r'(documentation|evidence) of.+?(HIV|human immunodeficiency virus)',
    r'(HIV|human immunodeficiency virus).+?(HAART|retroviral).+?',
    r'(known )?human immunodeficiency virus \(HIV\) infection',
    r'(known )?infection with (HIV|human immunodeficiency virus)',
    r'known.+?(HIV|human immunodeficiency virus)',
    r'diagnosis of (HIV|human immunodeficiency virus) infection',
    r'(HIV|human immunodeficiency virus).+?infections?',
    r'infect.+?(HIV|human immunodeficiency virus)',
    r'positiv.+(HIV|human immunodeficiency virus)',
    r'(HIV|human immunodeficiency virus)(-| )positiv',
    r'risk of.+(HIV|human immunodeficiency virus)',
    r'immunodeficiency[A-Z0-9 -,]+(HIV|human immunodeficiency virus)',
    r'patients (who have|with).+(HIV|human immunodeficiency virus)',
    r'clinically (evident|significant).+(HIV|human immunodeficiency virus)',
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
for x in REGEXES:
    print(x)


def get_true_hiv_status(conn, id):
    c = conn.cursor()
    c.execute("SELECT hiv_eligible FROM hiv_status WHERE NCTId=?", [id])
    result = c.fetchone()
    if result is None:
        return annotate_interactive(conn, id)
    else:
        return result[0]


def score_text(counter, text):
    chunks = re.split(".{,15}(inclusion|exclusion).{,15}$", text, flags=re.MULTILINE | re.IGNORECASE)
    score = 0
    multiplier = 1
    for blk in chunks:
        blk = blk.strip()
        if 'inclusion' in blk.lower():
            multiplier = 1
            print("[INCLUSION BLOCK]")
        elif 'exclusion' in blk.lower():
            multiplier = -1
            print("[EXCLUSION BLOCK]")
        for l in re.split(r'\n+|[A-Z ]+: +|[A-Z0-9]{4,}\. +', blk, flags=re.MULTILINE | re.IGNORECASE):
            l = l.strip()
            if l:
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
                        print("[%s, %s] (%s): %s" % (counter, s, rx, l))
                        break
                if not matched:
                    print("[%s, UNKNOWN] %s" % (counter, l))
    print(score, score >= 0)
    return int(score >= 0)


if __name__ == '__main__':
    print("Signatures: %s" % len(REGEXES))

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    true_scores = []
    predicted_scores = []

    if len(sys.argv) == 3 and sys.argv[2] == 'annotated':
        c.execute(
            "SELECT t1.NCTId, t1.EligibilityCriteria FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId")
    else:
        c.execute("SELECT NCTId, EligibilityCriteria FROM studies ORDER BY random() LIMIT 10")

    counter = 1
    for row in c.fetchall():
        print(row[0])
        true_scores.append(int(get_true_hiv_status(conn, row[0])))
        predicted_scores.append(score_text(counter, row[1]))
        counter += 1

    mismatches = []
    for i in range(len(true_scores)):
        if true_scores[i] != predicted_scores[i]:
            mismatches.append(i + 1)
    print("Count:   : %s" % len(true_scores))
    print("Incorrect: %s" % str(mismatches))
    print("Accuracy : %s" % accuracy_score(true_scores, predicted_scores))
    print(classification_report(true_scores, predicted_scores, target_names=['HIV-ineligible', 'HIV-eligible']))
    print("AUC:     : %s" % roc_auc_score(true_scores, predicted_scores))
    print("Confusion matrix:")
    print(confusion_matrix(true_scores, predicted_scores))
