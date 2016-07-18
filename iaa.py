#!/usr/bin/env python3

from collections import defaultdict
import csv
import numpy as np
import sys

from sklearn import metrics


RANGES = (
    (121, 150),
    (241, 270),
    (361, 390),
    (481, 510),
    (601, 630),
    (721, 750),
    (841, 870),
    (961, 990),
    (1081, 1110),
    (1201, 1230),
    (1321, 1350),
    (1441, 1470)
)

value_map = {
    'Ineligible': 0,
    'Indeterminate': 1,
    'Eligible (conditionally)': 2,
    'Eligible (unconditionally)': 3,
}

if __name__ == '__main__':
    data = {}
    # populate data
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)
        header = next(reader)
        categories = header[2:-1]
        for category in categories:
            data[category] = defaultdict(list)
        for row in reader:
            id = int(row[0])
            for x in zip(categories, row[2:-1]):
                x = list(x)
                x[1] = value_map.get(x[1])
                if x[1] is not None:
                    data[x[0]][id].append(x[1])
    # # only keep ids that contain two annotations
    # for category in categories:
    #     dc = data[category]
    #     for id in dc.keys():
    #         if len(dc[id]) != 2:
    #             del dc[id]
    # split and calculate
    for category in categories:
        dc = data[category]
        kappas = []
        for r in RANGES:
            r = range(r[0], r[1] + 1)
            y1 = [dc[id][0] for id in r if len(dc[id]) == 2]
            y2 = [dc[id][1] for id in r if len(dc[id]) == 2]
            assert(len(y1) == len(y2))
            if len(y1) > 0:
                ck = metrics.cohen_kappa_score(y1, y2)
                print("%s %s y1: %s" % (category, r, y1))
                print("%s %s y2: %s" % (category, r, y2))
                print("%s %s kappa (%d): %f" % (category, r, len(y1), ck))
                kappas.append(ck)
        mean_kappa = np.mean(kappas)
        print("%s mean: %f" % (category, mean_kappa))



