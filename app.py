import streamlit as st

st.title("Approximation de Taylor") #titre
st.write("Bienvenue dans votre outil de visualisation.") #description

st.subheader('Calculer formule de Taylor') #sous titre

#1.2
st.latex(r"Pn(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k") #Pn

#1.3 Le script est exécuté plusieurs fois

#3.1

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt
import numpy

def taylor1(X, derivatives):
    def taylor_base(x):
        a = 0
        for i in range(len(derivatives)): #pour chaque dérivée en 0...
            a+= derivatives[i]*(x**i)/factorial(i) #on calcule la sommme de taylor
        return a
    return taylor_base(X)

#3.2

def taylor2(X, f, n):
    derivatives = []
    a = f
    for i in range(n+1):
        derivatives.append(a(0.0)) #on calcule et ajoute chaque dérivée en 0
        a = grad(a) #on dérive un rang au dessus à chaque tour
    return taylor1(X,derivatives)

#3.3

def custom(X):
    return 2*np.sin(X) + np.cos(X/4)

#4.2

FONCTIONS = {"sin" : numpy.sin, "cos" : numpy.cos, "exp" : numpy.exp, "ln(1+x)": lambda x: numpy.log(1 + x),"cust" : custom}
#dico classique

#4.3

with st.sidebar:
    f = st.selectbox("Choisir une fonction",list(FONCTIONS.keys())) #FONCTIONS.keys pour
    #accéder aux clés du dico qui sont les noms donnés aux fonctions mathématiques
    n = st.slider("Ordre n", min_value=0, max_value=10, value=3) #défaut = value
    largeur = st.slider("Largeur de l'intervalle", min_value=2.0, max_value=20.0,value=12.6)


#4.4

f = FONCTIONS[f] #on récupère la fonction choisie

X = np.linspace(-largeur/2,largeur/2,400)

Y = f(X)

Y_taylor = taylor2(X,f,n)

fig, ax = plt.subplots()
ax.plot(X, Y, label="f(x)")
ax.plot(X, Y_taylor, label="f(x) Taylor")
st.pyplot(fig) #tracé des deux courbes pour comparer

#La qualité de l'approximation quand on s'éloigne de 0 dimiinue.
#Pour ln(x+1) au delà de abs(x)= 1, quand on augmente n, la
#qualité ne s'améliore pas.


#5.1

def taux_coincidence(Y, Y_taylor, seuil):
    return np.mean(np.abs(Y - Y_taylor) < seuil) #mean donne la moyenne des True=1 et False=0

seuil = st.sidebar.select_slider("Seuil", options=[10**-6, 10**-5, 10**-4, 10**-3, 10**-2],value=10**-3)

st.write("Taux de coïncidence :", taux_coincidence(Y, Y_taylor, seuil))

#Une comparaison avec des nan renvoie toujours False, donc le taux
#devient plus bas qu'il ne devrait l'être

#5.2

afficher_erreur = st.checkbox("Afficher l'erreur")

if afficher_erreur : #si la case est cochée alors :
    erreur = np.abs(Y - Y_taylor)
    fig, ax = plt.subplots()
    ax.plot(X, erreur)
    ax.set_yscale("log") #échelle logarithmique
    ax.set_xlabel("x")
    ax.set_ylabel("|f(x) - Pn(x)|")
    ax.set_title("Erreur")
    st.pyplot(fig)

if "ordres" not in st.session_state: #si on n'a pas encore exécuté
    st.session_state.ordres = []

if st.button("Ajouter cet ordre à la comparaison"):
    if n not in st.session_state.ordres: #pour éviter les doublons
        st.session_state.ordres.append(n)

if st.button("Réinitialiser"):
    st.session_state.ordres = [] #efface et remplace par une liste vide

fig, ax = plt.subplots()
ax.plot(X, Y, label="f(x)") #on trace f

for n in st.session_state.ordres : #pour chaque ordre coché :
    Y_k = taylor2(X, f, n) #on calcule le Taylor
    ax.plot(X, Y_k, label=f"Taylor ordre {n}") #on trace les taylor

ax.legend() #on affiche les légendes
st.pyplot(fig) #on affiche