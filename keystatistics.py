"""Program carries out PCA (dimensional reduction) followed by LDA on Physics abstracts.
   Abstracts are classified into PARTICLE (0), OPTICAL (2) or ASTRO (3)
   Documents should be in directory of this program and have extension ".keys".
   
   Each document should contain the Classification in the form:
   CLASSIFICATION: XXX  where XXX is PARTICLE, OPTICAL or ASTRO
   
   Each document should contain  s title in the form:
   TITLE: XXX  where XXX is the title.
   
   Each word is on a separate line. common non-technical words were filtered out by separate program (keygen)
   
"""
import glob
import numpy as np
import math
from numpy import linalg as LA
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

num_pca = 8

phys_id = {}
phys_id["PARTICLE"] = 0
phys_id["OPTICAL"]  = 1
phys_id["ASTRO"]    = 2
phys_id["GR"]       = 3

for (num_abstracts, f) in enumerate(glob.glob("*.keys"), 1):
    pass
    
print("num abstracts ", num_abstracts)

titles = ["" for x in range(num_abstracts)]
classification = ["" for x in range(num_abstracts)]

dict_set = defaultdict(set)
for f in glob.glob("*.keys"):
   with open(f, 'r') as fin:
       for line in fin:
      
           if -1 != line.find("TITLE"):
               id = int(f.lstrip("paper_").rstrip(".txt.keys")) -1
               titles[id] = line.replace("TITLE:", "").rstrip()
               
           elif -1 != line.find("CLASSIFICATION"):
               id = int(f.lstrip("paper_").rstrip(".txt.keys")) -1
               classification[id] = line.replace("CLASSIFICATION:", "").rstrip().lstrip()
           else:
               w = line.rstrip()
               dict_set[w].add(f) 
           
                  
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
        

#PCA via SVD
U, s, VT = LA.svd(X, full_matrices = False)

# Eigenvectors are columns of U
count = 0  
for w in dict_set:
   if len(dict_set[w]) > 1:
      print( "{0:20} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f}".format(w, U[count][0], U[count][1], U[count][2], U[count][3]) )
      count = count + 1
print("")

classdict = {}
classdict["PARTICLE"] = np.zeros(num_pca)
classdict["OPTICAL"] = np.zeros(num_pca)
classdict["ASTRO"] = np.zeros(num_pca)
classdict["GR"] = np.zeros(num_pca)


# Scores are s * rows of VT
count  = 0
for i in range(num_abstracts):
    print("{0:30} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f} {5: 2.4f}".format(titles[i] ,s[0]*VT[0][i], s[1]*VT[1][i], s[2]*VT[2][i], s[3]*VT[3][i], s[4]*(VT[4][i]) ) )
    nv = np.zeros(num_pca)
    for j in range(num_pca):
        nv[j] = s[0]*VT[j][i]
    nv = nv /LA.norm(nv) 
    if count == 0:
        X = nv
        Y = phys_id[classification[i]]
    else:
        X = np.vstack((X, nv))
        Y = np.hstack((Y, phys_id[classification[i]]))
    count  = count + 1

    print(i, " ", titles[i]) 
    classdict[classification[i]] = np.add(classdict[classification[i]], nv)     

print("")
print("PARTICLE ", classdict["PARTICLE"])
print("")
print("OPTICAL ", classdict["OPTICAL"])
print("")
print("ASTRO ", classdict["ASTRO"])
print("")
print("GR ", classdict["GR"])
print("")

# This prints out dot products of each PCA reduced vector with mean of that class
for i in range(num_abstracts):
    nv = np.zeros(num_pca)
    for j in range(num_pca):
        nv[j] = s[0]*VT[j][i]
    d0 = np.dot(nv, classdict["PARTICLE"])
    d0 = d0/(LA.norm(nv) * LA.norm(classdict["PARTICLE"]) )
    d1 = np.dot(nv, classdict["OPTICAL"])
    d1 = d1/(LA.norm(nv) * LA.norm(classdict["OPTICAL"]) )
    d2 = np.dot(nv, classdict["ASTRO"])
    d2 = d2/(LA.norm(nv) * LA.norm(classdict["ASTRO"]) )
    d3 = np.dot(nv, classdict["GR"])
    d3 = d3/(LA.norm(nv) * LA.norm(classdict["GR"]) )
    print(titles[i], d0, d1, d2, d3 )
  
#LDA
clf = LinearDiscriminantAnalysis()
clf.fit(X, Y)
print("")
print("Original Class labels:")
print(Y)
print("")
print("Predicted (LDA) Class labels:")
Z = clf.predict(X)
print(Z)
print("")
print("Misclassified documents:")
for i in range(num_abstracts):
    if Y[i] != Z[i]:
        print(titles[i])
        
#tSNE visualisation
X_embedded = TSNE(n_components=2, n_iter = 100000, init = 'pca').fit_transform(X)
for i in range(num_abstracts):
    print(Y[i], X_embedded[i][0], X_embedded[i][1] )
    
fig = plt.figure(figsize=(8,8))

x  = np.transpose(X_embedded)[0]
y  = np.transpose(X_embedded)[1]
c  = Y

plt.scatter(x, y, c = c)
plt.show()


