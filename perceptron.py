import numpy as np

m = np.zeros((10))
order = [9]

X = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]])
theta = np.array([0,0])
theta_0 = 0
Y = np.array([[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1]])


for t in range(20):
    if(t == 0):
        for i in order:
            if (Y[i] * ((np.dot(theta, X[i]) + theta_0)) <= 0):
                theta = theta + Y[i] * X[i]
                theta_0 = theta_0 + Y[i]
                m[i] = m[i] + 1
    else:
        for i in range(10):
            if (Y[i] * ((np.dot(theta, X[i]) + theta_0)) <= 0):
                theta = theta + Y[i] * X[i]
                theta_0 = theta_0 + Y[i]
                m[i] = m[i] + 1

print(m)
print(theta, theta_0)
