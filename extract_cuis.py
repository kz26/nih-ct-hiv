#!/usr/bin/env python3

import json
import os
import sys
import xml.etree.ElementTree as ET

def extract_cuis(filename):
    """
    Returns a a list of the CUIs extracted from the study's MetaMap XML file
    """
    with open(filename) as f:
        f.readline()  # workaround for non-XML first line
        root = ET.fromstring(f.read())
        cui_lines = []

        for phrase in root.findall('.//Phrase'):
            cuis = []
            mapping = phrase.find('.//Mapping')  # use first mapping only
            if mapping:
                for candidate in mapping.findall('.//Candidate'):
                    cui = candidate.find('CandidateCUI').text
                    sem_types = set([st.text for st in candidate.findall('.//SemType')])
                    if sem_types & {'aapp', 'dsyn', 'fndg', 'lbpr', 'lbtr', 'moft', 'phsu', 'topp', 'virs'}:
                        if int(candidate.find('Negated').text) == 1:
                            cui = 'N' + cui
                        cuis.append(cui)
            if len(cuis):
                cui_lines.append('\n'.join(cuis))
        return cui_lines

if __name__ == '__main__':
    path = sys.argv[1]
    data = {}
    i = 0
    for fn in os.listdir(path):
        study_id, ext = os.path.splitext(fn)
        if ext.lower() == '.xml':
            i += 1
            sys.stderr.write(str(i) + ' ' + study_id + '\n')
            data[study_id] = extract_cuis(os.path.join(path, fn))
    print(json.dumps(data))

