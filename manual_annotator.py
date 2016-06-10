#!/usr/bin/env python3

import argparse
import os
import sqlite3
import sys
import webbrowser
from time import sleep


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
        c.execute('SELECT NCTId, BriefTitle, Condition, StudyType, EligibilityCriteria FROM studies \
                  WHERE NOT EXISTS(SELECT * FROM hiv_status \
                                   WHERE studies.NCTId=hiv_status.NCTId) ORDER BY random()')
        for row in c.fetchall():
            self.prompt_for_annotation(row[0], [row[1], row[2].replace('\n', ', '), row[3], row[4]], allow_skip=True)
            sleep(0.5)
            os.system('clear')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar='FILE', dest='db_path', default='studies.sqlite',
                        help='SQLite database file to use')
    subparsers = parser.add_subparsers(dest='subcmd', title='subcommand')
    subparsers.required = True

    parser_interactive = subparsers.add_parser('interactive',
                                               help='annotate unlabeled studies from the database interactively')
    ns = parser.parse_args()
    db = Database(ns.db_path)
    if ns.subcmd == 'interactive':
        db.annotate_interactive()


