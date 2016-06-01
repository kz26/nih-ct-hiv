#!/usr/bin/env python3

import sqlite3
import sys
import xml.etree.ElementTree as ET


if __name__ == '__main__':

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    tree = ET.parse(sys.argv[2])
    root = tree.getroot()

    first_row = tree.find('row')
    col_schema = []
    for cell in first_row.iter('cell'):
        schema = [cell.get('column'), 'text']
        if schema[0] == 'NCTId':
            schema.append('PRIMARY KEY')
        col_schema.append(schema)
    c.execute("CREATE TABLE IF NOT EXISTS studies (%s)" % ', '.join([' '.join(x) for x in col_schema]))
    counter = 0
    for row in tree.iter('row'):
        values = []
        for cell in row.iter('cell'):
            values.append(cell.text)
        assert(len(values) == len(col_schema))
        placeholder = ','.join('?' * len(col_schema))
        c.execute("INSERT OR REPLACE INTO studies VALUES(%s)" % placeholder, values)
        counter += 1
    print("Added/updated %s records" % counter)
    conn.commit()
    conn.close()



