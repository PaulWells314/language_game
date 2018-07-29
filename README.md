There are two progranms.
keygen.py generates a list of technical keywords from a text document.
keystatistics.py carries out PCA and LDC on training and test data produced with keygen.py. t-SNE is also used to visualise training data.

Use:

python keygen.py paper_72.txt

This generates a file called paper_72.txt.keys

python keystatistics.py

This carries out learning on all files in same directory with name of form paper_xxx.txt.keys
Keys file should have lines such as following:

TITLE: Galaxy_correlation

CLASSIFICATION: UNKNOWN

CLASSIFICATION can be PARTICLE, OPTICAL, ASTRO, GR, or UNKNOWN. UNKNOWN inicates test data without label.
