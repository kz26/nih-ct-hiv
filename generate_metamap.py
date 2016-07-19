#!/usr/bin/env python3

import os
import sqlite3
import sys



if __name__ == '__main__':

    database = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, 0o0755)

    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute(
        'SELECT NCTId FROM annotations')
    for row in c.fetchall():
        study_id = row[0]
        out_file = os.path.join(output_dir, study_id + '.xml')
        if not os.path.exists(out_file):
            print("./manual_annotator.py -f %s print --ec-only --ascii %s | \
metamap --XMLf --prune 35 --blanklines 0 --negex > %s" % (database, study_id, out_file))
