import numpy as np
from numpy.linalg import inv
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import scipy.stats as stats
import scipy.special
import time
import datetime
import matplotlib.image as mpimg


#Closing all figures
plt.close('all')

#To time the file
start = time.time()

'Loading the data and converting values into floats'
images = scipy.io.loadmat("hw1data.mat")
inputdata = images["X"]
inputlabels = images["Y"]

N, pixels = np.shape(inputdata)

ind = np.arange(np.size(inputlabels))
shuffled = np.random.shuffle(ind)

ind = ind.astype(int)

'Splitting the data into training sets of different sizes'
train_test_split = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9500]

test_inputs = np.float64(inputdata)[ind[9499:-1],:]
test_labels = np.float64(inputlabels)[ind[9499:-1],0]

training_indices = ind[0:train_test_split[-1]]

'Generating training data'
train_inputs = np.float64(inputdata[training_indices,:])
train_labels = np.float64(inputlabels[training_indices,0])

passes = 3

"TRAIN - V0"
classes = np.unique(train_labels)
w0 = np.zeros(pixels)
w0foreachclass = []
T = passes*np.size(train_labels)
for k in range(0, len(classes)):
    for t in range(0, T):
        i = np.mod(t, np.size(train_labels))
        xi = train_inputs[i,:]
        
        if train_labels[i] == classes[k]:
            yi = 1.0
        else:
            yi = -1.0
            
        if yi*np.dot(w0,xi)  <= 0.0:
            w0 = w0 + yi*xi
        else:
            w0 = w0
    w0foreachclass.append(w0)

 
"CLASSIFY - V0"  
classifications = np.zeros(np.size(test_labels))
for p in range(0, np.size(test_labels)):
    argmaxarr = np.zeros(len(w0foreachclass))
    for k in range(0, len(w0foreachclass)):       
        argmaxarr[k] = np.dot(w0foreachclass[k],test_inputs[p,:])   
    classifications[p] = np.argmax(argmaxarr)
 
err0 = np.float32(np.count_nonzero(classifications-test_labels))
err0 = err0/np.size(test_labels)    
        

"TRAIN - VI"
w0foreachclassVI = []
classes = np.unique(train_labels)
w0 = np.zeros(pixels)

yclass = []
for k in range(0, len(classes)):
    ytest = np.zeros(np.size(train_labels))
    for j in range(0, np.size(train_labels)):
        if train_labels[j] == classes[k]:
            ytest[j] = 1.0
        else:
            ytest[j] = -1.0
    yclass.append(ytest)        

T = passes*np.size(train_labels)
for k in range(0, len(classes)):
    for t in range(0, T):
        alldotproducts = np.zeros(np.size(train_labels))
        for j in range(0, np.size(train_labels)):
            alldotproducts[j] = np.dot(yclass[k][j]*w0, train_inputs[j,:])
               
        i = np.argmin(alldotproducts)
        xi = train_inputs[i,:]
            
        if train_labels[i] == classes[k]:
            yi = 1.0
        else:
            yi = -1.0
            
        if yi*np.dot(w0,xi)  <= 0.0:
            w0 = w0 + yi*xi
        else:
            w0 = w0
            break
    print(k)
        
    w0foreachclassVI.append(w0)



"CLASSIFY - VI" 
classificationsVI = np.zeros(np.size(test_labels))
for p in range(0, np.size(test_labels)):
    argmaxarr = np.zeros(len(w0foreachclassVI))
    for k in range(0, len(w0foreachclassVI)):       
        argmaxarr[k] = np.dot(w0foreachclassVI[k],test_inputs[p,:])   
    classificationsVI[p] = np.argmax(argmaxarr)
    
err1 = np.float32(np.count_nonzero(classificationsVI-test_labels))
err1 = err1/np.size(test_labels)    


"TRAIN - V2"
classes = np.unique(train_labels)
w0foreachclassV2 = []
c = []
ks = []

w0 = np.zeros(pixels)
T = passes*np.size(train_labels)

for k in range(0, len(classes)):
    ka = 1
    wkforeachclass=[]
    ck = [0]
    for t in range(0, T):
        i = np.mod(t, np.size(train_labels))
        xi = train_inputs[i,:]
    
        if train_labels[i] == classes[k]:
            yi = 1.0
        else:
            yi = -1.0
             
        if yi*np.dot(w0,xi)  <= 0.0:
            w0 = w0 + yi*xi
            ck.append(1)
            ka = ka + 1
            wkforeachclass.append(w0)
            
        else:
            ck[ka-1] = ck[ka-1] + 1
        
            
    w0foreachclassV2.append(wkforeachclass)
    c.append(ck)
    ks.append(ka)       
    


"CLASSIFY - V2" 

classificationsV2 = np.zeros(np.size(test_labels))
for p in range(0, np.size(test_labels)):
    argmaxarr = np.zeros(len(w0foreachclassV2))
    for q in range(0, len(w0foreachclassV2)):
        dots = np.zeros(np.int(ks[q]))
        for k in range(0, np.int(ks[q])-1):
            dots[k] = c[q][k]*np.sign(np.dot(w0foreachclassV2[q][k], test_inputs[p,:]))
            #dots[k] = dots[k]
        argmaxarr[q] = np.sign(np.sum(dots))
#        print(k)
        
    classificationsV2[p] = np.argmax(argmaxarr)

err2 = np.float32(np.count_nonzero(classificationsV2-test_labels))
err2 = err2/np.size(test_labels) 

print(err0)
print(err1)
print(err2)



