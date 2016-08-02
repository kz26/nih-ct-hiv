# Requirements

* Python 3.5.x+
* scikit-learn 0.17.1+
* matplotlib 1.5.1+

# Explanation of files/directories

config/
Contains JSON configuration files for each of the various parameters and datasets for the machine learning models.

cui/
Contains files describing the MetaMap CUIs found for each dataset. These Python pickle files are generated from running extract_cuis.py on a directory of MetaMap XML output files.

generate_metamap.py
A script that reads all NCTIds from the annotations table of a SQLite database and generates a batch shell script for running MetaMap on the eligibility criteria.

iaa.py
Used to calculate interannotator agreement from a specially formatted CSV file.

manual_annotator.py
Multipurpose program/script used to interactively annotate random studies as well as other things like printing the eligibility criteria. Mostly used now in conjunction with generate_metamap.py

ml_classify.py
The implementation of the ML and ML+NER algorithms. Takes one argument, which is the path to a JSON configuration file.

re_classify.py
The implementation of the rule/regex-based classifier. Works only for HIV.

se2sqlite.py
Used to import an XML dump of studies from Russell's SE4 tool into a SQLite database. Run with -h to see command line syntax.

studies.sqlite
Contains the study data and HIV annotations for the cancer-HIV dataset.

studies_cs.sqlite
Contains the study data and crowdsourced HIV, pregnancy annotations for the "most recent" dataset.

studies_hiv_merged.sqlite
Created by merging studies.sqlite and studies_cs.sqlite (HIV annotations only).

annotations/
Contains the crowdsourced annotation data used to create studies_cs.sqlite. set_master.xlsx is created by concatenating all the individual
set[X].xlsx spreadsheets and serves as input to iaa.py (need to first replace the string labels with numeric values and then save as CSV.)

Any other content not mentioned here is obsolete and is not likely to be useful.


# Extending to other diseases/conditions

1. Obtain an XML dump of the studies of interest using SE4.
2. Load XML dump into a new SQLite database using se2sqlite.py.
3. Create an table named "annotations" using the following schema: "NCTId" foreign key to studies.NCTId, and
one integer column for each disease/condition status being annotated (e.g. hiv, pregnancy.)
4. Come up with an annotation scheme and annotate a sufficient amount of eligibility criteria (500-1000).
5. Create a config file.
6. Run ./ml_classify.py <config_file>.  Model parameters will probably need to be tweaked for optimal performance.

Trained models can be exported by defining the "export" option in a configuration file. This model can then be
used in other scenarios using the "import" option.
