import numpy as np


x = np.array([[1,1,-1],[1,-1,0],[-1,0,1]])
y = np.array([[1],[-1],[1]])
theta = np.array([[1,-2,0],[3,-1,1],[4,2,-3]])
theta_0 = np.array([[1],[2],[3]])
loss = 0

k = len(y)

for i in range(k):
    z = y[i] * (np.matmul(theta[i].transpose(), x[i]) + theta_0[i])
    if (z >= 1):
        loss = loss + 0
    else:
        loss = loss + (1 - z)


#print (y[0] * (np.matmul(theta[0].transpose(), x[0]) + theta_0[0]))
# print (loss/k)
# print (x.shape[1])

file = np.loadtxt("F:\Micromasters - stat & data science\project 1\project1\sentiment_analysis\stopwords.txt", dtype=str)
print (file)
