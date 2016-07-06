#!/usr/bin/env python3

import argparse
from collections import OrderedDict
from datetime import datetime
import random
import sqlite3
import sys
import xml.etree.ElementTree as ET

def convert_ct_start_date(ds):
    """Convert a ClinicalTrials.gov start date to ISO 8601 YYYY-MM-DD"""
    return datetime.strptime(ds, "%B %Y").strftime("%Y-%m-%d")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='category', default='main',
                        help='a category to associate with the study records being added')
    parser.add_argument('-p', dest='probability', type=float, default=1.0,
                        help='selection probability from 0.0-1.0')
    parser.add_argument('input_xml', help='path to the input XML file')
    parser.add_argument('output_db', help='path to the output database file')

    ns = parser.parse_args()

    tree = ET.parse(ns.input_xml)
    root = tree.getroot()

    col_schema = OrderedDict()
    for child in root.find('STUDY'):
        col_schema[child.tag] = 'text'
        if child.tag == 'NCTId':
            col_schema[child.tag] += ' PRIMARY KEY'
    col_schema['category'] = 'text'
    print(col_schema)

    conn = sqlite3.connect(ns.output_db)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS studies (%s)" % ', '.join([' '.join([k, col_schema[k]]) for k in col_schema]))
    counter = 0
    for study in root.findall('STUDY'):
        if random.random() > 1 - ns.probability:
            values = OrderedDict()
            for child in study:
                if child.tag == 'StartDate':
                    values[child.tag] = convert_ct_start_date(child.text)
                elif child.tag in values:
                    values[child.tag] += '\n' + child.text
                else:
                    values[child.tag] = child.text
            values['category'] = ns.category
            if set(values.keys()) == set(col_schema.keys()):
                placeholder = ','.join('?' * len(col_schema.keys()))
                try:
                    c.execute("INSERT INTO studies VALUES(%s)" % placeholder, tuple(values.values()))
                except sqlite3.IntegrityError as e:
                    print(values.get('NCTId', '') + ' ' + str(e))
                else:
                    counter += 1
                    print(counter)
            else:
                print("Schema mismatch: " + values.get('NCTId', ''))
    print("Added/updated %s records" % counter)
    conn.commit()
    conn.close()



