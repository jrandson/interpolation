'''
diag: Return the diagonal (or off-diagonal) elements of a square matrix as a 1D array, or convert a 1D array into a square
matrix with zeros on the off-diagonal
dot: Matrix multiplication
trace: Compute the sum of the diagonal elements
det: Compute the matrix determinant
eig: Compute the eigenvalues and eigenvectors of a square matrix
inv: Compute the inverse of a square matrix
pinv: Compute the Moore-Penrose pseudo-inverse inverse of a square matrix
qr: Compute the QR decomposition
svd: Compute the singular value decomposition (SVD)
solve: Solve the linear system Ax = b for x, where A is a square matrix
lstsq: Compute the least-squares solution to y = Xb'''

from interpolacao import *
from numpy.random import randn


def f(x):
    return np.sin(0.1*x)

def df(x):
    return np.cos(0.1*x)*0.1

def code1():
    x = range(6)
    y = [1,3,7,13,21,30]

    interp = Interpolacao(x,y)
    w0,w1 = interp.exponencial()
    print w0,w1
    y_ = w0*pow(w1,x)

    plt.plot(x,y,'r*')
    plt.plot(x,y_,'b--')
    plt.show()

def code2():
    x = range(5)
    y = np.array([1.0, 3.0, 7.0, 13.0, 21.0])

    interp = Interpolacao(x,y)

    w0,w1 = interp.regression_grad()
    #w0,w1 = interp.exponencial()

    y_ = np.array(x)*w0 + w1
    #y_ = w1*pow(w0,x)
    plt.plot(x,y, 'ro')
    plt.plot(x,y_, 'bo--')
    plt.show()

def code3():
    x = np.arange(60)
    #y = np.array([1.0, 3.0, 7.0, 13.0, 21.0])
    #y = np.array([19.01, 3.99, -1.00, 4.01, 18.99, 45.00])
    y = randn(60).cumsum()

    interp = Interpolacao(x,y)

    #w0, w1, w2 = interp.ordem2()
    w0, w1, w2, w3, w4, w5 = interp.regressao(5)
    #w0,w1 = interp.exponencial()

    #y_ = w1*x + w0
    #y_ = w2*x**2 + w1*x + w0
    y_ = w5*x**5 + w4*x**4 + w3*x**3 + w2*x**2 + w1*x + w0
    #y_ = w1*pow(w0,x)
    plt.plot(x,y, 'ro')
    plt.plot(x,y_, '--')
    plt.show()

code3()
