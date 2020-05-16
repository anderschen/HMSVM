import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
#Data set

x_neg=np.array([[3,4],[1,4],[2,3]])
x_pos=np.array([[6,-1],[7,-1],[5,-3],[2,4]]) 
y_neg=np.array([-1,-1,-1])
y_pos=np.array([1,1,1,1])

x=np.vstack((x_neg,x_pos))
y=np.concatenate((y_neg,y_pos))

C=10
m,n=x.shape
y=y.reshape(-1,1)*1.
x_dash=y*x
H=np.dot(x_dash,x_dash.T)*1.

P=cvxopt_matrix(H)
q=cvxopt_matrix(-np.ones((m,1)))
G=cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
h=cvxopt_matrix(np.hstack((np.zeros(m),np.ones(m)*C)))
A=cvxopt_matrix(y.reshape(1,-1))
b=cvxopt_matrix(np.zeros(1))
sol=cvxopt_solvers.qp(P,q,G,h,A,b)
alphas=np.array(sol['x'])

#computing qp 

w=((y*alphas).T@x).reshape(-1,1)
S=(alphas>1e-4).flatten()
b=y[S]-np.dot(x[S],w)

#Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])
xs=np.linspace(0,10,50)
fig = plt.figure(figsize = (10,10))
plt.scatter(x_neg[:,0], x_neg[:,1], marker = 'x', color = 'r', label = 'Negative -1')
plt.scatter(x_pos[:,0], x_pos[:,1], marker = 'o', color = 'b',label = 'Positive +1')
plt.grid(color='gray', linestyle='-', linewidth=0.5,alpha=0.5)
plt.xlim(0,10)
plt.ylim(-5,5)
plt.plot(xs,-(w[0]/w[1])*xs-b[0]/w[1],color='r')
plt.show()


