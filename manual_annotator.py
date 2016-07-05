#!/usr/bin/env python3

import argparse
import os
import re
import requests
import sqlite3
import string
import subprocess
import sys
import webbrowser
from time import sleep
import xml.etree.ElementTree as ET

API_URL = "https://clinicaltrials.gov/ct2/show/%s?displayxml=true"

class Database(object):
    def __init__(self, path):
        super().__init__()
        self.conn = sqlite3.connect(path)
        self.counter = 0

    def save_annotation(self, id, value, commit=True):
        self.conn.execute("INSERT OR REPLACE INTO hiv_status VALUES(?, ?)", [id, value])
        if commit:
            self.conn.commit()

    def prompt_for_annotation(self, id, content, allow_skip=False):
        print('-' * 40)
        print(id)
        print('\n'.join(content))
        print('-' * 40)
        v = None
        if allow_skip:
            prompt = 'Enter y/n/i/b/s --> '
        else:
            prompt = 'Enter y/n/i/b/s --> '
        self.print_status()
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
        self.save_annotation(id, v)
        value = v
        self.counter += 1
        return value

    def print_status(self):
        c = self.conn.cursor()
        c.execute('SELECT COUNT(*) FROM hiv_status')
        total = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM hiv_status WHERE hiv_eligible=2')
        eligible = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM hiv_status WHERE hiv_eligible=1')
        implicit = c.fetchone()[0]
        print("Annotated %s this session, %s total, %s eligible, %s implicit" %
              (self.counter, total, eligible, implicit))

    def annotate_interactive(self):
        c = self.conn.cursor()
        c.execute("SELECT NCTId, BriefTitle, Condition, StudyType, EligibilityCriteria FROM studies \
                  WHERE StudyType LIKE '%Interventional%' AND NOT EXISTS(SELECT * FROM hiv_status \
                                   WHERE studies.NCTId=hiv_status.NCTId) ORDER BY random()")
        for row in c.fetchall():
            self.prompt_for_annotation(row[0], (row[1], row[2].replace('\n', ', '), row[3], row[4]), allow_skip=True)
            sleep(0.5)
            os.system('clear')

    def cherry_pick(self, study_id):
        r = requests.get(API_URL % study_id)
        root = ET.fromstring(r.text)
        values = [None, None, None, study_id]
        values.append(root.find('overall_status').text)  # values[4]
        values.append(root.find('brief_title').text)
        values.append(root.find('condition').text)
        values.append(root.find('intervention').find('intervention_name').text)
        values.append(root.find('eligibility').find('criteria').find('textblock').text)
        values.append(root.find('study_type').text)
        plc = ', '.join(['?'] * len(values))
        self.conn.execute("INSERT OR REPLACE INTO studies VALUES(%s)" % plc, values)
        self.prompt_for_annotation(study_id,
                                   (values[5], values[6].replace('\n', ', '), values[7], values[4], values[8]))
        self.print_status()

    def print(self, study_id, ec_only=False, print_ascii=False, raw=False):
        c = self.conn.cursor()
        c.execute(
            'SELECT t1.BriefTitle, t1.Condition, t1.EligibilityCriteria \
             FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=? AND t1.NCTId=t2.NCTId ORDER BY t1.NCTId',
            [study_id])
        row = c.fetchone()
        if raw:
            if ec_only:
                text = row[2]
            else:
                text = '\n'.join(row)
        else:
            title, condition, ec = row
            if ec_only:
                lines = []
            else:
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
            text = '\n'.join(lines)
        if print_ascii:
            cp = subprocess.run(['iconv', '-t', 'ascii//TRANSLIT'], input=text, stdout=subprocess.PIPE,
                                universal_newlines=True)
            text = cp.stdout
        print(text)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar='FILE', dest='db_path', default='studies.sqlite',
                        help='SQLite database file to use')
    subparsers = parser.add_subparsers(dest='subcmd', title='subcommand')
    subparsers.required = True

    parser_interactive = subparsers.add_parser('interactive',
                                               help='annotate unlabeled studies from the database interactively')

    parser_cherry_pick = subparsers.add_parser('cherry-pick', help='add and annotate a study from ClinicalTrials.gov')
    parser_cherry_pick.add_argument('study_id', help='NCTID identifier from ClinicalTrials.gov')

    parser_print = subparsers.add_parser('print', help='output the title, condition, and EC of a study')
    parser_print.add_argument('study_id', help='NCTID identifier from ClinicalTrials.gov')
    parser_print.add_argument('--ec-only', help='print eligibility criteria only', action='store_true')
    parser_print.add_argument('--ascii', help='convert to ASCII', action='store_true')
    parser_print.add_argument('--raw', help='output as-is without any processing', action='store_true')

    ns = parser.parse_args()
    db = Database(ns.db_path)
    if ns.subcmd == 'interactive':
        db.annotate_interactive()
    elif ns.subcmd == 'cherry-pick':
        db.cherry_pick(ns.study_id)
    elif ns.subcmd == 'print':
        db.print(ns.study_id, ec_only=ns.ec_only, print_ascii=ns.ascii, raw=ns.raw)


