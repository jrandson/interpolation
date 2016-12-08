#!/usr/bin/python

import matplotlib.pyplot as plt
from random import random
import numpy as np
from numpy.linalg import inv, solve
from numpy.random import randn

class Interpolacao(object):
    """docstring for """
    def __init__(self, x,y):
        self.n = len(x)
        self.X = np.array(x)
        self.Y = np.array(y)


    def ordem1(self):
        n = len(self.X)
        soma_x = sum(self.X)
        soma_x2 = sum(self.X**2)
        soma_y = sum(self.Y)
        soma_xy = sum(self.X*self.Y)

        A = np.array([[n,soma_x],[soma_x,soma_x2]])
        B = np.array([[soma_y],[soma_xy]])
        coefs = inv(A).dot(B) #solve(A,B)

        return coefs # y = w1*x + w0

    def ordem2(self):
        n = len(self.X)
        soma_x = sum(self.X)
        soma_x2 = sum(self.X**2)
        soma_x3 = sum(self.X**3)
        soma_x4 = sum(self.X**4)
        soma_y = sum(self.Y)
        soma_xy = sum(self.X*self.Y)
        soma_x2y = sum(self.Y*self.X**2)

        A = np.array([[n,soma_x, soma_x2],[soma_x,soma_x2, soma_x3],[soma_x2, soma_x3, soma_x4]])
        B = np.array([[soma_y],[soma_xy],[soma_x2y]])
        coefs = inv(A).dot(B) #solve(A,B)

        return coefs # w0, w1, w2

    #does a regression of level n
    def regressao(self,n=1):
        A = []
        for i in range(n+1):
            row = []
            for j in range(i,n+i+1):
                tmp = sum(self.X**j)
                row.append(tmp)
            A.append(row)

        B = []
        for i in range(n+1):
            B.append([sum(self.X**i * self.Y)])

        A = np.array(A)
        B = np.array(B)

        coefs = inv(A).dot(B) #solve(A,B)
        print coefs
        return coefs

    #interpolacao iterativa
    def regression_grad(self):

        step = 0.001
        w = np.array([.0, .0])
        i = 0
        E = self.Y - (w[0]*self.X + w[1])
        while np.mean(E) > 0.01:
            g1 = sum(self.Y - (w[0] + w[1]*self.X))
            g2 = sum(self.Y - (w[0] + w[1]*self.X)*self.X)

            grad = np.array([g1,g2])
            w += 2*step*grad
            i += 1
            E = self.Y - (w[0]*self.X + w[1])
        print i
        return w

    def exponencial(self):
        #y = a*b^x
        y = np.log(self.Y)
        coefs = self.ordem1()
        self.Y = np.exp(y)

        w0,w1 = coefs
        a = np.exp(w0)
        b = np.exp(w1)

        return [a,b]

    def quadratica(self):
        x = self.X
        y = self.Y

        step = 0.01
        w = np.array([.0, .0, .0])
        i = 0
        E = y - (w[0]*x**2 + w[1]*x + w[2])

        while abs(sum(E)) > 0.01:
            g1 = sum(y - (w[0]*x**2 + w[1]*x + w[2])*x**2)
            g2 = sum(y - (w[0]*x**2 + w[1]*x + w[2])*x)
            g3 = sum(y - (w[0]*x**2 + w[1]*x + w[2]))

            grad = np.array([g1, g2, g3])
            print grad
            w += 2*step*grad
            i += 1
            E = y - (w[0]*x**2 + w[1]*x + w[2])
        print w
        return w

    def optimize(self, f,df, x0, ni = .01):
        ni = .01
        T = 200
        opt = f(x0)
        while t < T:
            if df(x) > 0:
                opt += -ni*df(x)
            elif df(x) < 0:
                opt += ni*df(x)
            x += .1
            t += 1

        return (x,opt)



#-------------------------------------------------------

if __name_ == '__main__':
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
