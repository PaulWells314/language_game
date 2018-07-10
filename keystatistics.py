import glob
import numpy as np
import math
from numpy import linalg as LA

titles = ["" for x in range(25)]

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
           else:
               id = int(f.lstrip("paper_").rstrip(".txt.keys")) -1
               titles[id] = line.replace("TITLE:", "").rstrip()
                  

count = 0
for w in dict_set:
    if len(dict_set[w]) > 1:
        vec = np.zeros(25)
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

count = 0
for w in dict_set:
   if len(dict_set[w]) > 1:
      print( "{0:20} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f}".format(w, U[count][0], U[count][1], U[count][2], U[count][3]) )
      count = count + 1
print("")


for i in range(25):
    print("{0:30} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f} {5: 2.4f}".format(titles[i] ,s[i]*VT[i][0], s[i]*VT[i][1], s[i]*VT[i][2], s[i]*VT[i][3], s[i]*(VT[i][4]) ) )
