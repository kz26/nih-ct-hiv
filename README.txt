# Requirements

* Python 3.5.x+
* scikit-learn 0.17.1+
* matplotlib 1.5.1+

# Usage

The best-performing model is contained in ml_mm_classify.py. To run, execute the script with no arguments:
./ml_mm_classify.py

You will need the study data and annotations, found in studies.sqlite, and the corresponding set of MetaMap CUIs,
found in cuis.pickle (a binary Python pickle).

# Extending for other use cases

The repository contains other support scripts (extract_cuis.py, se2sqlite.py, manual_annotator.py) that are used
for various data conversion tasks that you may find useful in preparing your own data and annotations.