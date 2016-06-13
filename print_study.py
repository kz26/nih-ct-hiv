#!/usr/bin/env python3

import re
import sqlite3
import string
import subprocess
import sys

DATABASE = 'studies.sqlite'


def filter_study(title, condition, ec):
    """take one study and returns a filtered version with only relevant lines included"""
    lines = [title + '.']
    for l in condition.split('\n'):
        lines.append(l + '.')
    segments = re.split(
        r'\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|(?:[A-Z][A-Za-z]+ )+?[A-Z][A-Za-z]+: +|; +| (?=[A-Z][a-z])',
        ec, flags=re.MULTILINE)
    for i, l in enumerate(segments):
        l = l.strip()
        if l:
            if l:
                if ' ' in l and l[-1] not in string.punctuation:
                    l += '.'
                lines.append(l)
    return '\n'.join(lines)


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        'SELECT t1.BriefTitle, t1.Condition, t1.EligibilityCriteria \
         FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=? AND t1.NCTId=t2.NCTId ORDER BY t1.NCTId', [sys.argv[1]])
    row = c.fetchone()
    out = filter_study(*row)
    subprocess.run(['iconv', '-t', 'ascii//TRANSLIT'], input=out, universal_newlines=True)
    print()