#!/usr/bin/env python3

import os
import sqlite3
import sys
import webbrowser
from time import sleep


def _save_annotation(c, id, value):
    c.execute("INSERT OR REPLACE INTO hiv_status VALUES(?, ?)", [id, value])


def annotate_interactive(conn, id, counter, allow_skip=False):
    c = conn.cursor()
    c.execute("SELECT BriefTitle, StudyType, EligibilityCriteria FROM studies WHERE NCTId=?", [id])
    row = c.fetchone()
    print('-' * 40)
    print(id)
    print('\n'.join(row))
    print('-' * 40)
    v = None
    if allow_skip:
        prompt = 'Enter y/n/i/b/s --> '
    else:
        prompt = 'Enter y/n/i/b/s --> '
    print_status(conn, counter)
    while v is None:
        rv = input(prompt).lower()
        if rv == 'y':
            v = 2
        elif rv == 'n':
            v = 0
        elif rv == 'i':
            v = 1
        elif rv == 'b':
            v = None
            webbrowser.open("https://clinicaltrials.gov/ct2/show/%s" % id)
        elif allow_skip and rv == 's':
            return None
    _save_annotation(c, id, v)
    value = v
    conn.commit()
    return value


def print_status(conn, counter):
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM hiv_status')
    total = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM hiv_status WHERE hiv_eligible=2')
    eligible = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM hiv_status WHERE hiv_eligible=1')
    implicit = c.fetchone()[0]
    print("Annotated %s this session, %s total, %s eligible, %s implicit" % (counter, total, eligible, implicit))

if __name__ == '__main__':
    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    c.execute('SELECT NCTId FROM studies WHERE NOT EXISTS(SELECT * FROM hiv_status WHERE studies.NCTId=hiv_status.NCTId) ORDER BY random()')
    i = 0
    for row in c.fetchall():
        annotate_interactive(conn, row[0], i, allow_skip=True)
        i += 1
        sleep(0.5)
        os.system('clear')
