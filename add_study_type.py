#!/usr/bin/env python3

import requests
import sqlite3
import xml.etree.ElementTree as ET


API_URL = "https://clinicaltrials.gov/ct2/show/%s?displayxml=true"
DATABASE = "studies.sqlite"


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT NCTId FROM studies WHERE StudyType IS NULL")
    counter = 0
    for row in c.fetchall():
        study_id = row[0]
        r = requests.get(API_URL % study_id)
        tree = ET.fromstring(r.text)
        study_type = tree.find('study_type').text
        conn.execute("UPDATE studies SET StudyType=? WHERE NCTId=?", [study_type, study_id])
        counter += 1
        print("%s %s" % (counter, study_type))
    conn.commit()
    conn.close()