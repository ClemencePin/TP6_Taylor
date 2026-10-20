import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")

st.subheader('Calculer formule de Taylor')

#1.2
st.latex(r"Pn(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k")

#1.3 Le script est exécuté plusieurs fois

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt
import numpy

def taylor1(X, derivatives):
    def taylor_base(x):
        a = 0
        for i in range(len(derivatives)):
            a+= derivatives[i]*(x**i)/factorial(i)
        return a
    return taylor_base(X)

def taylor2(X, f, n):
    derivatives = []
    a = f
    for i in range(n+1):
        derivatives.append(a(0.0))
        a = grad(a)
    return taylor1(X,derivatives)

def custom(X):
    return 2*np.sin(X) + np.cos(X/4)

FONCTIONS = {"sin" : numpy.sin, "cos" : numpy.cos, "exp" : numpy.exp, "ln(1+x)": lambda x: numpy.log(1 + x),"cust" : custom}

with st.sidebar:
    f = st.selectbox("Choisir une fonction",list(FONCTIONS.keys()))
    n = st.slider("Ordre n", min_value=0, max_value=10, value=3)
    largeur = st.slider("Largeur de l'intervalle", min_value=2.0, max_value=20.0,value=12.6)

f = FONCTIONS[f]

X = np.linspace(-largeur/2,largeur/2,400)

Y = f(X)

Y_taylor = taylor2(X,f,n)

fig, ax = plt.subplots()
ax.plot(X, Y, label="f(x)")
ax.plot(X, Y_taylor, label="f(x) Taylor")
st.pyplot(fig)

