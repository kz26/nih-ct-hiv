#!/usr/bin/env python3

import os
import sqlite3

DATABASE = 'studies.sqlite'
OUTPUT_DIR = 'metamap_out'

if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        'SELECT t1.NCTId FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')
    for row in c.fetchall():
        study_id = row[0]
        out_file = os.path.join(OUTPUT_DIR, study_id + '.xml')
        if not os.path.exists(out_file):
            print("./manual_annotator.py print --ec-only --ascii %s | \
metamap --XMLf --prune 35 --blanklines 0 --negex > %s" % (study_id, out_file))
