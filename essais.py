import numpy
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

X = np.linspace(0,2*numpy.pi,100)

sin_der = np.vectorize(grad(numpy.sin))
X_der = sin_der(X)

print(X_der)
print(np.cos(X)) #c'est bon les deux sont égaux

#3.1

from math import factorial

def taylor1(X, derivatives):
    def taylor_base(x):
        a = 0
        for i in range(len(derivatives)):
            a+= derivatives[i]*(x**i)/factorial(i)
        return a
    return taylor_base(X)

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)

plt.plot(X, taylor1(X, sinus10))
plt.plot(X, np.sin(X))
plt.show()

#3.2

def taylor2(X, f, n):
    derivatives = []
    a = f
    for i in range(n+1):
        derivatives.append(a(0.0))
        a = grad(a)
    return taylor1(X,derivatives)

assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)

plt.plot(X,Y)
plt.plot(X,Y2)
plt.show()

def verif(Y1,Y2):
    a = 0
    for i in range(len(X)):
        if numpy.abs(Y1[i]-Y2[i]) < 10**(-3):
            a+=1
    return (a/len(X))*100
print(verif(Y,Y2)) #résultat : 42 %

plt.plot(X,np.cos(X))
plt.plot(X,taylor2(X, np.cos,0))
plt.show() #je constate que l'approx au degré 0 est une droite : logique

#3.3

def custom(X):
    return 2*np.sin(X) + np.cos(X/4)
Y3 = custom(X)
Y4 = taylor2(X,custom,14)

print(verif(Y3,Y4)) #résultat : 61,5 %

plt.plot(X,Y3)
plt.plot(X,Y4)
plt.show()





