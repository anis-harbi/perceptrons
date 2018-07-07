import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import time

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

     
c = 0               # Hyperparameter. When c= 0, kernel funciton is homogeneous
def kernelFunc(x,y,degree,c):
    return np.power(np.dot(x,y)+c,degree)

errwrtd = []

for degree in [5,6,7,8,9,10]:           # Degree of kernel funciton
    # TRAIN - (iii) Kernel
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
                
            if yi*kernelFunc(w0,xi,degree,c) <= 0.0:
                w0 = w0 + yi * xi
            else:
                w0 = w0
        w0foreachclass.append(w0)
    
    
    "CLASSIFY - (iii) Kernel"
    classifications = np.zeros(np.size(test_labels))
    for p in range(0, np.size(test_labels)):
        argmaxarr = np.zeros(len(w0foreachclass))
        for k in range(0, len(w0foreachclass)):
            argmaxarr[k] = kernelFunc(w0foreachclass[k],test_inputs[p,:],degree,c)
            classifications[p] = np.argmax(argmaxarr)

    err0 = np.float32(np.count_nonzero(classifications-test_labels))
    err0 = err0/np.size(test_labels)
    errwrtd.append(err0)

plt.plot(range(5,11),errwrtd)

plt.show()