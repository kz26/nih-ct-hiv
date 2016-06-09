#!/usr/bin/env python3

import sqlite3


DATABASE = 'studies.sqlite'


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        'SELECT t1.NCTId FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')
    for row in c.fetchall():
        study_id = row[0]
        print("./print_study %s | metamap --XMLf --blanklines 0 --negex > metamap_out/%s.xml" % (study_id, study_id))