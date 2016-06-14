#!/usr/bin/env python3
# Returns a term count matrix in pickle format

from collections import defaultdict
import os
import pickle
import sqlite3
import sys
import xml.etree.ElementTree as ET

from sklearn.feature_extraction import DictVectorizer

DATABASE = 'studies.sqlite'
METAMAP_XML_DIR = 'metamap_out'


def xml_to_features(study_id):
    with open(os.path.join(METAMAP_XML_DIR, study_id + '.xml')) as f:
        f.readline()  # workaround for non-XML first line
        root = ET.fromstring(f.read())
        features = defaultdict(int)
        names = {}
        for x in root.find('.//Negations').findall('.//NegConcept'):
            ncui = 'N' + x.find('NegConcCUI').text
            features[ncui] += 1
            names[ncui] = '[N] ' + x.find('NegConcMatched').text
        for mapping in root.findall('.//Mapping[1]'):
            for candidate in mapping.findall('.//Candidate'):
                cui = candidate.find('CandidateCUI').text
                features[cui] += 1
                names[cui] = candidate.find('CandidatePreferred').text
                names['N' + cui] = '[N] ' + candidate.find('CandidatePreferred').text
        return (features, names)


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    study_ids = []
    cui_names = {}
    def gen_features(sids, cn):
        counter = 0
        c.execute('SELECT t1.NCTId FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')
        for row in c.fetchall():
            study_id = row[0]
            sids.append(study_id)
            features, names = xml_to_features(study_id)
            cn.update(names)
            counter += 1
            sys.stderr.write(str(counter) + '\n')
            yield features

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(gen_features(study_ids, cui_names))
    data = {
        'vectorizer': vectorizer,
        'study_ids': study_ids,
        'cui_names': cui_names,
        'X': X
    }
    pickle.dump(data, sys.stdout.buffer)

