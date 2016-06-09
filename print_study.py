#!/usr/bin/env python3

import sqlite3
import subprocess
import sys


DATABASE = 'studies.sqlite'


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        'SELECT t1.BriefTitle, t1.Condition, t1.EligibilityCriteria \
         FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=? AND t1.NCTId=t2.NCTId ORDER BY t1.NCTId', [sys.argv[1]])
    row = c.fetchone()
    out = '\n'.join(row)
    subprocess.run(['iconv', '-t', 'ascii//TRANSLIT'], input=out, universal_newlines=True)
    print()