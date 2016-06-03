#!/usr/bin/env python3

import sqlite3


def _save_annotation(c, id, value):
    c.execute("INSERT OR REPLACE INTO hiv_status VALUES(?, ?)", [id, value])


def annotate_interactive(conn, id):
    c = conn.cursor()
    c.execute("SELECT EligibilityCriteria FROM studies WHERE NCTId=?", [id])
    print(c.fetchone()[0])
    v = None
    while v is None:
        rv = input('Enter y/n --> ').lower()
        if rv == 'y':
            v = True
        elif rv == 'n':
            v = False
    _save_annotation(c, id, v)
    value = v
    conn.commit()
    return value
