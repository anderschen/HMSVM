import numpy as np 
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

#Data set
x_neg = np.array([[3,4],[1,4],[2,3]])
y_neg = np.array([-1,-1,-1])
x_pos = np.array([[6,-1],[7,-1],[5,-3]])
y_pos = np.array([1,1,1])

#行並排
#
x=np.vstack((x_pos,x_neg))
y=np.concatenate((y_pos,y_neg))
print(x)
m,n=x.shape
y=y.reshape(-1,1)*1.
x_dash=y*x
H=np.dot(x_dash,x_dash.T)*1.

#compute qp
P=cvxopt_matrix(H)
q=cvxopt_matrix(-np.ones((m,1)))
G=cvxopt_matrix(-np.eye(m))
h=cvxopt_matrix(np.zeros(m))
A=cvxopt_matrix(y.reshape(1,-1))
b=cvxopt_matrix(np.zeros(1))

#Setting solver parameters (change default to decrease tolerance) 

cvxopt_solvers.options['show_progress']=False
cvxopt_solvers.options['abstol']=1e-10
cvxopt_solvers.options['reltol']=1e-10
cvxopt_solvers.options['feastol']=1e-10

solve=cvxopt_solvers.qp(P,q,G,h,A,b)
alphas=np.array(solve['x'])


w=((y*alphas).T@x).reshape(-1,1)

S=(alphas>1e-4).flatten()
b=y[S]-np.dot(x[S],w)

print('Alpha=',alphas[alphas>1e-4])
print('w=',w.flatten())
print('b=',b[0])
xs=np.linspace(0,10,50)

fig=plt.figure(figsize=(10,10))
plt.scatter(x_neg[:,0],x_neg[:,1],marker='x',color='r',label='Negative -1')
plt.scatter(x_pos[:,0],x_pos[:,1],marker='o',color='b',label='Positive +1')
plt.grid(color='gray', linestyle='-', linewidth=0.5,alpha=0.5)


plt.grid(color='gray', linestyle='-', linewidth=0.5,alpha=0.5)
plt.xlim(0,10)
plt.ylim(-5,5)
plt.plot(xs,-(w[0]/w[1])*xs-b[0]/w[1],color='r')

plt.show()