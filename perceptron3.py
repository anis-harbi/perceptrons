import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

time0 = time.time()

#loads examples from file 
images = sio.loadmat("hw2data.mat")
inputdata = images["X"]
inputlabels = images["Y"]
X = np.array(inputdata)
Y = np.array(inputlabels)

np.random.seed(1)
dimentions = X.shape   
n = dimentions[0]       	#number of training examples
k =4 						#number of classes in intermediate layer
eta = .001 					#learning rate
iterations =10000 
alpha = 1.0					#inertia parameter, setting to 1 means it is not used
convergence = False 		#used fixed number of iterations instead

#initialize the weight parameters
W1 = 2*np.random.random((1,k)) - 1
W2 = 2*np.random.random((k,1)) - 1
b1 = 2*np.random.random((k,1)) - 1
b2 = np.random.random()

delta_W1 = 0
delta_W2 = 0
delta_b1 = 0
delta_b2 = 0

#defines the sigmoid function
def sigma(x,deriv=False):
	return 1.0/(1.0+1.0*np.exp(-x))

#repeat for a fixed number of iterations (until convergence)
for j in range(iterations):
	E = 0
	delta_W1 = 0
	delta_W2 = 0
	delta_b1 = 0
	delta_b2 = 0
	#print delta_W1

	for i in range(n):
		#load example (xi, yi)
		xi = X.item(i)
		yi = Y.item(i)
		#output of each layer
		o1 = sigma(xi*W1.T+b1)
		o2 = sigma(np.dot(W2.T,o1)+b2)
		#weight changea at (i-1)
		prev_delta_w1 =delta_W1
		prev_delta_w2 = delta_W2
		prev_delta_b1 =delta_b1
		prev_delta_b2 =delta_b2
		o2 = o2.item(0)
		#weight changes
		delta_W1 += xi*(o2 - yi)*o2*(1 - o2)*(W2.dot(o1.T.dot(1-o1))).T
		delta_W2 += (((o2 - yi)*o2*(1 - o2))*o1)
		delta_b1 += (o2 - yi)*o2*(1 - o2)*W2.dot(o1.T.dot(1-o1))
		delta_b2 += ((o2 - yi)*o2*(1 - o2))				
		#cumulative error
		E += ((o2-yi)**2)
	
	#update weight parameters with inertia (this reduces to normal update when alpha =1)
	W1 = W1 - (eta)*alpha*delta_W1 + (1-alpha)*prev_delta_w1
	W2 = W2 - (eta)*alpha*delta_W2 + (1-alpha)*prev_delta_w2
	b1 = b1 - (eta)*alpha*delta_b1 + (1-alpha)*prev_delta_b1
	b2 = b2 - (eta)*alpha*delta_b2 + (1-alpha)*prev_delta_b2

	if(j%100==0):
			print 'E(W1,W2,b1,b2) = ', E/n
	
#now that the parameters are learned
F = []
for i in range(n):
		xi = X.item(i)
		o1 = sigma(W1.T*xi+b1)
		o2 = sigma(np.dot(W2.T,o1)+b2)
		o2 = o2.item(0)
		F.append(o2)

print 'W1 =np.array(', W1, ')'
print 'W2 =np.array(', W2.T, ')'
print 'W2= W2.T'
print 'b1 = np.array(', b1.T, ')'
print 'b1 =b1.T'
print 'b2 =', b2
print (time.time() - time0)/60.0
plt.scatter(X,Y,c='blue')
plt.scatter(X,F, c='red')
plt.savefig('learned.png')

'''
#optimal weight parameters
W1 =np.array( [[ 1.67679114 , 2.07243296, -0.32613032, -0.61911229]] )
W2 =np.array( [[  2.4181115  , -0.21128199 ,-10.97472661,  14.1830294 ]] )
W2= W2.T
b1 = np.array( [[-5.57164615, -6.57660403, -1.42149693,  0.63914542]] )
b1 =b1.T
b2 = -2.25293867194
'''