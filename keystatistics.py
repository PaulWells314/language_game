"""Program carries out PCA (dimensional reduction) followed by LDA on Physics abstracts.
   Abstracts are classified into PARTICLE (0), OPTICAL (1), ASTRO (3), or GR(4)
   Documents should be in directory of this program and have extension ".keys".
   
   Each document should contain the Classification in the form:
   CLASSIFICATION: XXX  where XXX is PARTICLE, OPTICAL, ASTRO, or GR
   
   Each document should contain  s title in the form:
   TITLE: XXX  where XXX is the title.
   
   Each word is on a separate line. common non-technical words were filtered out by separate program (keygen)
   
"""
import sys
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

labels = ["PARTICLE", "OPTICAL", "ASTRO", "GR"]


# Reverse lookup
id_to_phys = {}
for j in phys_id:
    id_to_phys[phys_id[j]] = j

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
               print(id,classification[id] )
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
train_count = 0
test_count = 0
for i in range(num_abstracts):
    print("{0:30} {1: 2.4f} {2: 2.4f} {3: 2.4f} {4: 2.4f} {5: 2.4f}".format(titles[i] ,s[0]*VT[0][i], s[1]*VT[1][i], s[2]*VT[2][i], s[3]*VT[3][i], s[4]*(VT[4][i]) ) )
    nv = np.zeros(num_pca)
    for j in range(num_pca):
        nv[j] = s[0]*VT[j][i]
    nv = nv /LA.norm(nv) 
    if count == 0:
        X = nv
    else:
        X = np.vstack((X, nv))
        
    #Training data
    if classification[i] != 'UNKNOWN':
        if train_count == 0:        
            Y = phys_id[classification[i]]
            XTrain = nv
        else:
            Y = np.hstack((Y, phys_id[classification[i]]))
            XTrain = np.vstack((XTrain, nv))
        train_count = train_count + 1
        classdict[classification[i]] = np.add(classdict[classification[i]], nv)    
       
    #Test data     
    else:
        if test_count == 0:
            XTest = nv
        else:
            XTest = np.vstack((XTest, nv))
        test_count = test_count + 1
        
    count  = count + 1

    print(i, " ", titles[i]) 
     

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
    if classification[i] != 'UNKNOWN':
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
XTrain_new = clf.fit_transform(XTrain, Y)
print("")
print("Original Class labels:")
print(Y)
print("")
print("Predicted (LDA) Class labels on Training Set:")
Z = clf.predict(XTrain)
print(Z)
fig = plt.figure(figsize=(8,8))

x  = np.transpose(XTrain_new)[0]
y  = np.transpose(XTrain_new)[1]
c  = Y

plt.scatter(x, y, c = c)
plt.title("LDA of Physics abstracts")
plt.show()
print("")
print("Misclassified Training documents:")


for i in range(num_abstracts):
    if classification[i] != 'UNKNOWN':
        if Y[i] != Z[i]:
            print(titles[i])

print()         
print("Predictions of test data:")
j = 0
for i in range(num_abstracts):
    if classification[i] == 'UNKNOWN':
        print(titles[i], id_to_phys[clf.predict(XTest[j].reshape(1,-1))[0]] ) 
        j = j + 1       

print()

        
#tSNE visualisation
X_embedded = TSNE(n_components=2, n_iter = 100000, init = 'pca').fit_transform(XTrain)

fig = plt.figure(figsize=(8,8))
ax  = fig.add_subplot(1, 1, 1)

x  = np.transpose(X_embedded)[0]
y  = np.transpose(X_embedded)[1]
c  = Y

plt.scatter(x, y, c=c)
plt.title("t-SNE plot of Physics abstracts")
plt.show()

