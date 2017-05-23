"""
Démonstration du calcul de barycentres par Sinkhorn.


Utilise matplotlib, numpy et scipy.

"""


import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


## Fonctions

def norme(v):
    return np.sum(np.abs(v))/np.sum(np.ones_like(v))

def Gaussian(t,t0,sigma):
    return np.exp( -(t-t0)**2/(2*sigma**2) )

def normalize(p):
    return p/np.sum(p)

"""
P : array of square images
lbd : coefficient of each image in the barycenter
gamma : regularization strength
"""
def barycenter(P,lbd=None,gamma=0.1**2,iterations=100):
    global b,g,g2,xi
    K = np.size(P,2)
    N = np.size(P,0)
    n = 2*(N//2)+1 # width of the convolution kernel
    t = np.linspace(-n/(2*N),n/(2*N),n)
    g = normalize(np.exp(-t**2 / gamma)); g2 =  np.outer(g,g)
    #def xi(x):
    #    return convolve2d(x,g2,'same')+1e-19
    xi_div = 2
    sqrt_gamma = np.sqrt(gamma/xi_div)*N
    def xi(x):
        xi = np.copy(x)
        zeta = np.zeros_like(xi)
        for _ in range(xi_div):
            gaussian_filter(xi,sqrt_gamma,mode='constant',cval=0,truncate=20.0,output=zeta)
            xi[:] = zeta[:]
        #xi += 1e-19
        return xi

    if lbd is None:
        lbd = np.ones(K)/K
    b = np.ones((N,N,K)); a = np.ones((N,N,K))
    
    err = []
    for l in range(iterations):
        for k in range(K):
            a[:,:,k] = np.divide(P[:,:,k],xi(b[:,:,k]))
            
        log_q = np.zeros((N,N))
        for k in range(K):
            log_q += lbd[k] * np.log(np.maximum(1e-19, b[:,:,k] * xi(a[:,:,k])))
        q = np.exp(log_q)
        
        for k in range(K):
            b[:,:,k] = np.divide(q,xi(a[:,:,k]))
            
        err.append(np.array([norme(a[:,:,k] * xi(b[:,:,k]) - P[:,:,k]) for k in range(K)]))
        if np.any(np.isnan(a)):
            return [[0]]
    return q

def make_color_image(img,alpha,beta):
    m = np.min(img)
    M = np.max(img)
    gamma = 1-alpha-beta
    q = (img-m)/(M-m)
    n = np.size(q,0)
    q_color = np.zeros((n,n,3))
    q_color[:,:,0] = 1 - (1-alpha) * q
    q_color[:,:,1] = 1 - (1-beta) * q
    q_color[:,:,2] = 1 - (1-gamma) * q
    return q_color

def plot_interpol(rep1,rep2,number_of_subplots=5,gamma=0.05**2):
    n = len(rep1)
    rep1 = normalize(rep1); rep2 = normalize(rep2)
    P = np.zeros((n,n,2))
    P[:,:,0] = rep1; P[:,:,1] = rep2
    coefs = np.linspace(0,1,number_of_subplots)
    for i,alpha in enumerate(coefs):
        print("Calcul de l'interpolé",i+1,"/",number_of_subplots)
        lbd = np.array([alpha, 1-alpha])
        q = barycenter(P,lbd,gamma=gamma)
        ax1 = plt.subplot(1,number_of_subplots,i+1)
        ax1.axis('off')
        q_color = make_color_image(q,alpha,0)
        ax1.imshow(q_color)
    plt.show()
    

def plot_interpol3(rep1,rep2,rep3,subs=4,gamma=0.04**2):
    import matplotlib.gridspec as grsp
    n = len(rep1)
    rep1 = normalize(rep1); rep2 = normalize(rep2); rep3 = normalize(rep3)
    P = np.zeros((n,n,3))
    P[:,:,0] = rep1; P[:,:,1] = rep2; P[:,:,2] = rep3
    gs = grsp.GridSpec(subs,2*subs)
    total = subs*(subs+1)/2
    count = 0
    for i in range(subs):
        for j in range(subs-i):
            count+=1
            print("Calcul de l'interpolé %d/%d" % (count, total))
            alpha = i / (subs-1)
            beta = j / (subs-1)
            lbd = np.array([alpha, beta, 1-alpha-beta])
            q = barycenter(P,lbd,gamma=gamma)
            ax = plt.subplot(gs[i,i+2*j:i+2*(j+1)])
            #ax.get_xaxis().set_ticks([])
            #ax.get_yaxis().set_ticks([])
            ax.axis('off')
            q_color = make_color_image(q,alpha,beta)
            ax.imshow(q_color,interpolation='bicubic')
    plt.show()


## Formes

def circle(xx,yy,cx,cy,r):
    return np.array([[1.0*((x-cx)**2+(y-cy)**2 < r**2) for y in yy] for x in xx])

def one_circle(n):
    xx = np.arange(n+1); yy = np.arange(n+1)
    return circle(xx,yy,n/2,n/2,2*n/5)

def three_circles(n):
    xx = np.arange(n+1); yy = np.arange(n+1)
    return circle(xx,yy,n/4,n/4,n/6) + circle(xx,yy,3*n/4,3*n/4,n/6) + circle(xx,yy,3*n/4,n/4,n/6)
    
def letter_o(n):
    xx = np.arange(n+1); yy = np.arange(n+1)
    return circle(xx,yy,n/2,n/2,2*n/5) - circle(xx,yy,n/2,n/2,n/5)

def letter_t(n):
    nm = n/10; nM=9*n/10
    return [[ (nm<y<3*nm and nm<x<nM or (abs(x-n/2)<nm and nm<y<nM)) for x in range(n+1)] for y in range(n+1)]

def letter_n(n):
    nm = n/10; nM=9*n/10
    return [[ (nm<y<3*nm and nm<x<nM) or (abs(x-y)<2*nm and nm<x<nM and nm<y<nM) or (nM-2*nm<y<nM and nm<x<nM) for y in range(n+1)] for x in range(n+1)]

def square(n):
    nm = n/10; nM=9*n/10
    return [[ (nm<x<nM and nm<y<nM) for y in range(n+1)] for x in range(n+1)]

def empty_square(n):
    nm = n/10; nM=9*n/10
    a = 3*nm; b = n-3*nm
    return [[ (nm<x<nM and nm<y<nM) and not (a<x<b and a<y<b) for y in range(n+1)] for x in range(n+1)]

def heart(n):
    xx = np.linspace(-1.2,1.2,n+1)
    yy = np.linspace(1.4,-1,n+1)
    xx,yy = np.meshgrid(xx,yy)
    return 1.0*(xx**2 + (5*yy/4 - np.sqrt(np.abs(xx)))**2 <= 1)

shapes = [('Un disque','d',one_circle),('Trois disques','ddd',three_circles),('Un carré plein','c',square),('Un carré vide','cv',empty_square),('Un coeur','h',heart),('La lettre O','O',letter_o),('La lettre T','T',letter_t),('La lettre N','N',letter_n)]

## Partie interactive

import sys

print("""Ceci est un fichier de démonstration du calcul de barycentres par l'algorithme de Sinkhorn. Il permet d'afficher l'interpolation de différentes formes.""")
print("\n\nLes formes disponibles sont :\n")
format = "{:>15}{:>15}"
print(format.format("Forme","Code"))
print("-"*30)
for shape,code,fun in shapes:
    print(format.format(shape,code))
    
print("\nEntrez les codes des formes que vous voulez interpoler (pas plus de 3, séparées par des virgules) :")
codes = sys.stdin.readline()
sh = codes[:-1].split(',')
act = []
for s in sh:
    success=False
    for m,c,f in shapes:
        if s==c:
            act.append(f)
            success=True
            break
    if not success:
        print("Le code '{}' n'existe pas. Il sera ignoré.".format(s))

if len(act)<2 or len(act)>3:
    print("Choisissez deux ou trois formes valides.")
else:
    print("Choisissez la taille de l'image, en pixels (par défaut 50) :")
    img_size = sys.stdin.readline()[:-1]
    if img_size == "":
        img_size = 50
    else:
        img_size = int(img_size)
    print("Choisissez la valeur du coefficient de régularisation epsilon (par défaut 0.05, des valeurs plus petites donnent des images plus nettes mais augmentent l'instabilité):")
    epsilon = sys.stdin.readline()[:-1]
    if epsilon == "":
        epsilon = 0.05
    else:
        epsilon = float(epsilon)
    subdiv_default = 4 + (len(act) == 2)
    print("Choisissez le nombre de pas de subdivision dans l'affichage (par défaut {}) :".format(subdiv_default))
    subdiv = sys.stdin.readline()[:-1]
    if subdiv == "":
        subdiv = subdiv_default
    else:
        subdiv = int(subdiv)
    
    rep = [f(img_size) for f in act]
    if len(act) == 2:
        plot_interpol(rep[0],rep[1],number_of_subplots=subdiv,gamma=epsilon**2)
    else:
        plot_interpol3(rep[0],rep[1],rep[2],subs=subdiv,gamma=epsilon**2)
    
    
