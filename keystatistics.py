"""Program to calculate document eigenvectors and scores.
   Documents shoukd be in directory of this program and have extension ".keys"
"""
import glob
import numpy as np
import math
from numpy import linalg as LA
from collections import defaultdict

for (num_abstracts, f) in enumerate(glob.glob("*.keys"), 1):
    pass
    
print("num abstracts ", num_abstracts)

titles = ["" for x in range(num_abstracts)]

dict_set = defaultdict(set)
for f in glob.glob("*.keys"):
   with open(f, 'r') as fin:
       for line in fin:
           if -1 == line.find("TITLE"):
               w = line.rstrip()
               dict_set[w].add(f) 

           else:
               id = int(f.lstrip("paper_").rstrip(".txt.keys")) -1
               titles[id] = line.replace("TITLE:", "").rstrip()
                  
count = 0
for w in dict_set:
    if len(dict_set[w]) > 1:
        vec = np.zeros(num_abstracts)
        for d in dict_set[w]:
            id = int(d.lstrip("paper_").rstrip(".txt.keys"))
            vec[id-1] = 1

        if count == 0:
            X = vec
        else:
            X = np.vstack((X, vec))
        count = count = 1
        
m = X.mean(axis=0)
X = X -m

U, s, VT = LA.svd(X, full_matrices = False)

# Eigenvectors are columns of U
count = 0  
for w in dict_set:
   if len(dict_set[w]) > 1:
      print( "{0:20} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f}".format(w, U[count][0], U[count][1], U[count][2], U[count][3]) )
      count = count + 1
print("")

# Scores are s * rows of VT
for i in range(num_abstracts):
    print("{0:30} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f} {5: 2.4f}".format(titles[i] ,s[0]*VT[0][i], s[1]*VT[1][i], s[2]*VT[2][i], s[3]*VT[3][i], s[4]*(VT[4][i]) ) )

