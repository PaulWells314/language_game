import glob
import numpy as np
import math
from numpy import linalg as LA

dict_set = {}
for f in glob.glob("*.keys"):
   with open(f, 'r') as fin:
       for line in fin:
           if -1 == line.find("TITLE"):
               w = line.rstrip()
               if w in dict_set:
                  dict_set[w].add(f) 

               else:
                  dict_set[w] = set()
                  dict_set[w].add(f)                  
                
count = 0               
for w in dict_set:
    if len(dict_set[w]) > 1:
        vec = np.zeros(25)
        for d in dict_set[w]:
            id = int(d.lstrip("paper_").rstrip(".txt.keys"))
            vec[id-1] = 1
#       print(vec)
        if count == 0:
            X = vec
        else:
            X = np.vstack((X, vec))
        count = count = 1
        
m = X.mean(axis=0)
X = X -m

U, s, V = LA.svd(np.transpose(X), full_matrices = False)


count = 0
for w in dict_set:
   if len(dict_set[w]) > 1:
      print( "{0:20} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f}".format(w, V[0][count], V[1][count], V[2][count], V[3][count]) )
      count = count + 1

