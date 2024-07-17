import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def polynomial_kernel(X1, X2, c, p):
    kernel = np.matmul(X1, np.transpose(X2))
    kernel_matrix = (kernel + c) ** p
    return kernel_matrix
    raise NotImplementedError


X = np.array([[0,0],[2,0],[1,1],[0,2],[3,3],[4,1],[5,2],[1,4],[4,4],[5,5]])
Y = np.array([[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1]])
c = 1
d = 2
phi = np.zeros((10,3))
theta = [0,0,0]
theta0 = 0
m = np.zeros((10))
order = [7,8,9,6,1,0,2,4,3,5]

for i in range(10):
    phi[i] = [X[i][0]**2, np.sqrt(2)*X[i][0]*X[i][1], X[i][1]**2]



for t in range(80):
    if (t == 0):
        for i in order:
            if (Y[i] * ((np.dot(theta, phi[i]) + theta0)) <= 0):
                theta = theta + Y[i] * phi[i]
                theta0 = theta0 + Y[i]
                m[i] = m[i] + 1
    else:
        for i in range(10):
            if (Y[i] * ((np.dot(theta, phi[i]) + theta0)) <= 0):
                theta = theta + Y[i] * phi[i]
                theta0 = theta0 + Y[i]
                m[i] = m[i] + 1



print(theta, theta0)
print(m)





# svclassifier = SVC(kernel='poly', degree=2)
# svclassifier.fit(X, Y)
#
# plt.figure(figsize=(4, 3))
# plt.clf()
#
# plt.scatter(
#         svclassifier.support_vectors_[:, 0],
#         svclassifier.support_vectors_[:, 1],
#         s=80,
#         facecolors="none",
#         zorder=10,
#         edgecolors="k",
#     )
# plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors="k")
#
# plt.axis("tight")
# x_min = -1
# x_max = 6
# y_min = -1
# y_max = 6
#
# XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# Z = svclassifier.decision_function(np.c_[XX.ravel(), YY.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(XX.shape)
# plt.figure(figsize=(4, 3))
# plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
# plt.contour(
#         XX,
#         YY,
#         Z,
#         colors=["k", "k", "k"],
#         linestyles=["--", "-", "--"],
#         levels=[-0.5, 0, 0.5],
#     )
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
#
# plt.xticks(())
# plt.yticks(())
# plt.show()
#
# print(svclassifier.dual_coef_)
# print(svclassifier.intercept_)
