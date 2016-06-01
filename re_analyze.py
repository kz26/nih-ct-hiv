#!/usr/bin/env python3

import re
import sqlite3
import sys

from manual_annotator import annotate_interactive

from sklearn.metrics import precision_score, recall_score, f1_score

ABSOLUTE_POSITIVE_SIGNATURES = (
    r'(HIV|human immunodeficiency virus) testing is not required',
)

POSITIVE_ONLY_SIGNATURES = (
    r'(positive)?[A-Z0-9 -,]*?(HIV|human immunodeficiency virus)[A-Z0-9 -,]+?antibody',
)
POSITIVE_SIGNATURES = (
    r'seropositive for (HIV|human immunodeficiency virus)'
    r'any form of primary|secondary immunodeficiency',
    r'history of (HIV|human immunodeficiency virus)',
    r'History of primary|secondary immunodeficiency',
    r'(HIV|human immunodeficiency virus) antibody positive',
    r'known diagnosis of[A-Z -,]+?HIV/AIDS',
    r'test positive for[A-Z -,]+?(HIV|human immunodeficiency virus)',
    r'HIV \(Human Immunodeficiency Virus\) positive'
    r'documentation of[A-Z -,]+?(HIV|human immunodeficiency virus) infection',
    r'(known )?human immunodeficiency virus \(HIV\) infection',
    r'(known )?infection with (HIV|human immunodeficiency virus)',
    r'known[A-Z -,]+?(HIV|human immunodeficiency virus)',
    r'diagnosis of (HIV|human immunodeficiency virus) infection',
    r'(HIV|human immunodeficiency virus).+?infections?',
    r'infections?[A-Z -,]+?(HIV|human immunodeficiency virus)',
    r'asymptomatic (for)?[A-Z -,]+(HIV|human immunodeficiency virus)',
    r'positiv[A-Z -,]+(for)?[A-Z -,]+(HIV|human immunodeficiency virus)',
    r'(HIV|human immunodeficiency virus)(-| )positiv',
    r'HIV-seropositive',
    r'HIV infection',
    r'HIV\+',
)

NEGATIVE_SIGNATURES = (
    r'negative [A-Z0-9 -,]*?(HIV|human immunodeficiency virus)[A-Z0-9 -,]+?antibody',
    r'HIV-( +|$)',
)

ABSOLUTE_POSITIVE_REGEXES = [((re.compile(x, flags=re.IGNORECASE), True)) for x in ABSOLUTE_POSITIVE_SIGNATURES]
POSITIVE_ONLY_REGEXES = [((re.compile(x, flags=re.IGNORECASE), 1)) for x in POSITIVE_ONLY_SIGNATURES]
POSITIVE_REGEXES = [((re.compile(x, flags=re.IGNORECASE), 1)) for x in POSITIVE_SIGNATURES]
NEGATIVE_REGEXES = [((re.compile(x, flags=re.IGNORECASE), -1)) for x in NEGATIVE_SIGNATURES]
for x in POSITIVE_SIGNATURES:
    if 'positiv' in x:
        NEGATIVE_REGEXES.append((re.compile(x.replace('positiv', 'negativ'), flags=re.IGNORECASE), -1))
    NEGATIVE_REGEXES.append((re.compile(r'not? [A-Z0-9 -,]*?' + x, flags=re.IGNORECASE), -1))

REGEXES = ABSOLUTE_POSITIVE_REGEXES + NEGATIVE_REGEXES + POSITIVE_ONLY_REGEXES + POSITIVE_REGEXES


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
            cs = re.split('([A-Z ]+?: +)', l, flags=re.IGNORECASE)  # split by colons
            for l2 in cs:
                matched = False
                for rx, v in REGEXES:
                    if rx.search(l2):
                        matched = True
                        if v is True:
                            s = 1
                        else:
                            s = v * multiplier
                        score += s
                        print("[%s, %s] (%s): %s" % (counter, s, rx, l2))
                        break
                if not matched:
                    print("[%s, UNKNOWN] %s" % (counter, l2))
    print(score, score >= 0)
    return int(score >= 0)


if __name__ == '__main__':
    print("Signatures: %s" % len(REGEXES))

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    true_scores = []
    predicted_scores = []

    # c.execute("SELECT NCTId, EligibilityCriteria FROM studies ORDER BY random() LIMIT 10")
    c.execute(
        "SELECT t1.NCTId, t1.EligibilityCriteria FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId")
    counter = 1
    for row in c.fetchall():
        print(row[0])
        true_scores.append(int(get_true_hiv_status(conn, row[0])))
        predicted_scores.append(score_text(counter, row[1]))
        counter += 1

    print("True     : %s" % str(true_scores))
    print("Predicted: %s" % str(predicted_scores))
    print("Precision: %s" % precision_score(true_scores, predicted_scores))
    print("Recall   : %s" % recall_score(true_scores, predicted_scores))
    print("F score  : %s" % f1_score(true_scores, predicted_scores))
