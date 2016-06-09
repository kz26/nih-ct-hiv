#!/usr/bin/env python3

import sqlite3
import sys
import xml.etree.ElementTree as ET


if __name__ == '__main__':

    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    tree = ET.parse(sys.argv[2])

    first_row = tree.find('row')
    counter = 0
    for row in tree.iter('row'):
        values = []
        for cell in row.iter('cell'):
            if cell.get('column') == 'NCTId':
                c.execute("DELETE FROM studies WHERE NCTId=?", [cell.text])
                counter += 1
    print("Deleted %s records" % counter)
    conn.commit()
    conn.close()



