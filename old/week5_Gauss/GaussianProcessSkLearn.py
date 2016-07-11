import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcess

def f(x):
    return x*np.sin(np.pi*x)

numX = 3
lowX=-1
highX=1
thetaLow=1e-9
thetaHight=1.


X = np.random.uniform(lowX, highX,size = numX )
Y = f(X)
X=np.atleast_2d(X).T


noiseVariance =0# 1e-9


nugget=noiseVariance*np.ones(Y.size)


gp = GaussianProcess(corr='squared_exponential',theta0=0.5,thetaL=thetaLow,thetaU=thetaHight,nugget=nugget,random_start=5)



gp.fit(X,Y)

points =np.atleast_2d(  np.linspace(lowX, highX, 1000)).T

mean ,std = gp.predict(points, eval_MSE=True)


plt.plot(points,f(points), 'b-', linewidth =1)

plt.plot(points,mean, 'r-', linewidth =1)
plt.fill(np.concatenate([points, points[::-1]]),
        np.concatenate([mean - 2 * std,
                       (mean + 2* std)[::-1]]),
        alpha=.95, fc='grey', ec='None' )

plt.plot(X,Y, 'r.', markersize =10)
plt.show()
