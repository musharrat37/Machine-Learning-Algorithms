import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#from sklearn.svm import SVC

# we create 40 separable points
np.random.seed(0)
X = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]])
# Y = [0] * 5 + [1] * 5
#
# # fit the model
# clf = svm.SVC(kernel='linear')
# clf.fit(X, Y)
#
# # get the separating hyperplane
# w = clf.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(-1, 5)
# yy = a * xx - (clf.intercept_[0]) / w[1]
#
# # plot the parallels to the separating hyperplane that pass through the
# # support vectors
# b = clf.support_vectors_[0]
# yy_down = a * xx + (b[1] - a * b[0])
# b = clf.support_vectors_[-1]
# yy_up = a * xx + (b[1] - a * b[0])
#
# # plot the line, the points, and the nearest vectors to the plane
# plt.plot(xx, yy, 'k-')
# plt.plot(xx, yy_down, 'k--')
# plt.plot(xx, yy_up, 'k--')
#
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=80, facecolors='none')
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
#
# plt.axis('tight')
# plt.show()
#
# print (f"Decision Boundary (y ={w[0]}x1  + {w[1]}x2  {clf.intercept_[0] })")
# print(f'Margin : {1.0 /np.sqrt(np.sum(clf.coef_ ** 2)) }')



theta = np.array([0.5,0.5])
theta_0 = -2.5
labels = np.array([[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1]])
loss = 0

for i in range(10):
    z = labels[i] * (np.matmul(theta.transpose(), X[i]) + theta_0)
    if (z >= 1):
        loss = loss + 0
    else:
        loss = loss + (1-z)

print (loss)