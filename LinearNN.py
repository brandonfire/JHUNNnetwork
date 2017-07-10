import matplotlib.pyplot as plt
import numpy as np

 	
d = np.genfromtxt("Jdatacsv.csv",delimiter=',')
print (d[:,0] -27)/26
print d[:,1:51]
# Load the data and create the data matrices X and Y
# This creates a feature vector X with a column of ones (bias)
# and a column of car weights.
# The target vector Y is a column of MPG values for each car.
#X_file = np.genfromtxt('Jdatacsv.csv', delimiter=',', skip_header=1)
#N = np.shape(X_file)[0]
#X = np.hstack((np.ones(N).reshape(N, 1), X_file[:, 4].reshape(N, 1)))
#Y = X_file[:, 0]
#print Y
# Standardize the input
#X[:, 1] = (X[:, 1]-np.mean(X[:, 1]))/np.std(X[:, 1])
#print X
# There are two weights, the bias weight and the feature weight
#w = np.array([0, 0])#C Hu: need to have an array for the weights.
 
# Start batch gradient descent, it will run for max_iter epochs and have a step
# size eta
#max_iter = 100
#eta = 1E-3
#for t in range(0, max_iter):
    # We need to iterate over each data point for one epoch
#    grad_t = np.array([0., 0.])
#    for i in range(0, N):
#        x_i = X[i, :]
#        y_i = Y[i]
        # Dot product, computes h(x_i, w)
#        h = np.dot(w, x_i)-y_i
#        grad_t += 2*x_i*h
 
    # Update the weights
#    w = w - eta*grad_t
#print "Weights found:",w
 
# Plot the data and best fit line
#tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
#bf_line = w[0]+w[1]*tt
 
#plt.plot(X[:, 1], Y, 'kx', tt, bf_line, 'r-')
#plt.xlabel('Weight (Normalized)')

#plt.savefig('mpg.png')
 
#plt.show()
