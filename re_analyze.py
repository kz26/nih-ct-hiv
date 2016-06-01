#!/usr/bin/env python3

import re
import sqlite3
import sys

from manual_annotator import annotate_interactive

from sklearn.metrics import precision_score, recall_score

POSITIVE_SIGNATURES = (
    r'HIV-seropositive',
    r'HIV infection',
    r'seropositive for HIV'
    r'seropositive for human immunodeficiency virus',
    r'any form of primary|secondary immunodeficiency',
    r'History of primary|secondary immunodeficiency',
    r'HIV antibody positive',
    r'known diagnosis of[A-Z -,]+?HIV/AIDS',
    r'test positive for[A-Z -,]+?HIV',
    r'HIV (Human Immunodeficiency Virus) positive'
    r'documentation of[A-Z -,]+?HIV infection',
    r'(known )?human immunodeficiency virus (HIV) infection',
    r'(known )?infection with HIV',
    r'known[A-Z -,]+?HIV',
    r'diagnosis of HIV infection',
    r'HIV.+?infections?',
    r'infections?.+?HIV',
    r'HIV positiv',
    r'HIV\+'
)

NEGATIVE_SIGNATURES = (
    r'HIV-',
)

POSITIVE_REGEXES = [((re.compile(x, flags=re.IGNORECASE), 1)) for x in POSITIVE_SIGNATURES]
NEGATIVE_REGEXES = [((re.compile(x, flags=re.IGNORECASE), -1)) for x in NEGATIVE_SIGNATURES]
for x in POSITIVE_SIGNATURES:
    if 'positiv' in x:
        NEGATIVE_REGEXES.append((re.compile(x.replace('positiv', 'negativ'), flags=re.IGNORECASE), -1))
    NEGATIVE_REGEXES.append((re.compile(r'not? .*?' + x, flags=re.IGNORECASE), -1))

REGEXES = NEGATIVE_REGEXES + POSITIVE_REGEXES


def get_true_hiv_status(c, id):
    return annotate_interactive(c, id)


def score_text(counter, text):
    chunks = re.split("(^inclusion|exclusion).*$", text, flags=re.MULTILINE | re.IGNORECASE)
    score = 0
    multiplier = 1
    for blk in chunks:
        blk = blk.strip()
        if 'inclusion' in blk.lower():
            multiplier = 1
            print(blk)
            continue
        elif 'exclusion' in blk.lower():
            multiplier = -1
            print(blk)
            continue
        for l in re.split(r'\n+', blk):
            matched = False
            for rx, v in REGEXES:
                if rx.search(l):
                    matched = True
                    s = v * multiplier
                    score += s
                    print("[%s, %s] (%s): %s" % (counter, s, rx, l))
                    break
            if not matched:
                print("[%s, UNKNOWN] %s" % (counter, l))
    return int(score >= 0)


if __name__ == '__main__':
    print("Signatures: %s" % len(REGEXES))

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    true_scores = []
    predicted_scores = []

    #c.execute("SELECT NCTId, EligibilityCriteria FROM studies ORDER BY random() LIMIT 10")
    c.execute("SELECT t1.NCTId, t1.EligibilityCriteria FROM studies as t1, hiv_status as t2 WHERE t1.NCTId=t2.NCTId")
    counter = 1
    for row in c.fetchall():
        true_scores.append(int(get_true_hiv_status(conn, row[0])))
        predicted_scores.append(score_text(counter, row[1]))
        counter += 1

    print("True     : %s" % str(true_scores))
    print("Predicted: %s" % str(predicted_scores))
    print("Precision: %s" % precision_score(true_scores, predicted_scores))
    print("Recall   : %s" % recall_score(true_scores, predicted_scores))



