#!/usr/bin/env python3

import re
import sqlite3
import sys

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


def _save_annotation(c, id, value):
    c.execute("INSERT INTO hiv_status VALUES(?, ?)", [id, value])


def score_text(text):
    chunks = re.split(".{,15}(inclusion|exclusion).{,15}$", text, flags=re.MULTILINE | re.IGNORECASE)
    score = 0
    multiplier = 1
    for blk in chunks:
        blk = blk.strip()
        if 'inclusion' in blk.lower():
            multiplier = 1
        elif 'exclusion' in blk.lower():
            multiplier = -1
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
                        break
    return int(score >= 0)


if __name__ == '__main__':
    for x in REGEXES:
        print(x)

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    c.execute('SELECT NCTId, EligibilityCriteria FROM studies WHERE NOT EXISTS(SELECT * FROM hiv_status WHERE studies.NCTId=hiv_status.NCTId)')

    counter = 1
    for row in c.fetchall():
        score = score_text(row[1])
        if not score:
            _save_annotation(conn, row[0], score)
            conn.commit()
            counter += 1
    print("Added %s negative annotations" % counter)
