#!/usr/bin/env python3
# Returns a term count matrix in pickle format

import os
import pickle
import sqlite3
import string
import sys
import xml.etree.ElementTree as ET

from sklearn.feature_extraction.text import TfidfVectorizer

from print_study import filter_study

DATABASE = 'studies.sqlite'
METAMAP_XML_DIR = 'metamap_out'

REMOVE_PUNC = str.maketrans({key: None for key in string.punctuation})


def get_features(study_id):
    """
    Returns a CUI-concept mapping, and list of (CUI, start, len) features
    """
    with open(os.path.join(METAMAP_XML_DIR, study_id + '.xml')) as f:
        f.readline()  # workaround for non-XML first line
        root = ET.fromstring(f.read())
        features = {}
        names = {}

        # for x in root.find('.//Negations'):
        #     neg_type = x.find('NegType').text
        #     neg_trigger = x.find('.//NegTriggerPI')
        #     neg_pos = int(neg_trigger.find('StartPos').text)
        #     neg_length = int(neg_trigger.find('Length').text)
        #     features[neg_pos] = (neg_type, neg_length)
        #     names[neg_type] = neg_type
        #     ncui = x.find('.//NegConcCUI').text

        for phrase in root.findall('.//Phrase'):
            mappings = phrase.findall('.//Mapping[1]')
            if len(mappings):
                for mapping in mappings:
                    for candidate in mapping.findall('.//Candidate'):
                        cui = candidate.find('CandidateCUI').text
                        concept = candidate.find('CandidatePreferred').text
                        # sem_types = set([st.text for st in candidate.findall('.//SemType')])
                        if int(candidate.find('Negated').text) == 1:
                            cui = 'N' + cui
                            concept = '[N] ' + concept
                        names[cui] = concept
                        for pi in candidate.findall('.//ConceptPI'):
                            pos = int(pi.find('StartPos').text)
                            l = int(pi.find('Length').text)
                            features[pos] = ([cui], l)
            else:  # no mappings found, fall back to syntax units
                start_pos = int(phrase.find('PhraseStartPos').text)
                length = int(phrase.find('PhraseLength').text)
                tokens = []
                for su in phrase.findall('.//SyntaxUnit'):
                    pos = su.find('SyntaxType').text
                    if pos == 'punc':
                        pos = ' '
                    tokens.append(pos)
                features[start_pos] = (tokens, length)

        features_list = []
        for k, v in features.items():
            features_list.append((v[0], k, v[1]))
        features_list.sort(key=lambda x: x[1])

        return features_list, names


def features_to_text(features, text):
    """
    Replace text with features from the specified feature dictionary
    """

    i = 0
    orig_length = len(text)
    new_text = ''
    f = None
    while i < orig_length:
        if f is None:
            if len(features):
                f = features.pop(0)
            else:
                new_text += text[i:]
                break
        if i == f[1]:
            new_text += ' '.join(f[0]) + ' '
            i += f[2]
            f = None
        else:
            new_text += text[i:f[1]]
            i = f[1]

    return new_text



if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    study_ids = []
    cui_names = {}

    def gen_documents(sids, cn):
        counter = 0
        c.execute('SELECT t1.NCTId, t1.BriefTitle, t1.Condition, t1.EligibilityCriteria \
                   FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')
        for row in c.fetchall():
            study_id = row[0]
            sids.append(study_id)
            features, names = get_features(study_id)
            text = filter_study(*row[1:])
            text = features_to_text(features, text)
            # text = text.translate(REMOVE_PUNC)
            cn.update(names)
            counter += 1
            # sys.stderr.write(study_id + '\n')
            # sys.stderr.write(text + '\n')
            sys.stderr.write("[%s] %s\n" % (counter, study_id))
            yield text

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(gen_documents(study_ids, cui_names))
    data = {
        'vectorizer': vectorizer,
        'study_ids': study_ids,
        'cui_names': cui_names,
        'X': X
    }
    pickle.dump(data, sys.stdout.buffer)

