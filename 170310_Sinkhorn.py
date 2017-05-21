import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def norme(v):
    return np.sum(np.abs(v))/len(v)

"""
Calcule la distance d_M^lbd(r,c), approximation de facteur
lambda du transport optimal entre r et c.
r, c : vecteurs de même taille d
M : matrice d*d des distances
lambda : quand lambda -> +infini, la valeur approche
celle du transport optimal non régularisé
"""
def ot_sinkhorn_map_error(M,r,c,lbd=10,epsilon=1e-3,iter_bound=500):
    lbd /= len(r)
    initial_size = (len(r),len(c))
    I = (r>0)
    r = r[I]
    M = M[I,:]
    K = np.exp(-lbd * M)
    u = np.ones(len(r))/len(r)
    norm = 1
    prev_u = 0
    count = 0
    err_r = []; err_c = []
    #print("I",I,"\nr",r,"\nM",M,"\nK",K,"\nu",u,"\nK_",K_)
    while norm>epsilon and count<iter_bound:
        count+=1
        prev_u = u
        v = np.divide(c,np.transpose(K).dot(u))
        u = np.divide(r,K.dot(v))
        norm = norme(u-prev_u)
        # err_r.append(norme(K.dot(u) - r))
        err_r.append(norme(v * np.transpose(K).dot(u) - c))
        #print(u)
    if count>=iter_bound:
        print("Itercount exceeded")
    #plt.plot(np.log(err_r))
    #plt.plot(np.log(err_c))
    #plt.show()
    #plt.pause(5)
    #v = np.divide(c,np.transpose(K).dot(u))
    P = np.zeros(initial_size)
    P[I,:] = np.diag(u).dot(K).dot(np.diag(v))
    return P,err_r
    # return np.sum(np.multiply(u,np.multiply(K,M).dot(v)))

def ot_sinkhorn(*args):
    P,err = ot_sinkhorn_map_error(*args)
    return P

def ot_sinkhorn_value(M,*args):
    P = ot_sinkhorn(M,*args)
    return np.sum(np.multiply(M,P))

def distances(n,p=2):
    a = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            a[i,j] = abs(i-j)**p
    return a

def binomial(n,p=0.5):
    k = np.arange(n+1)
    return p**k * (1-p)**(n-k) * scipy.special.comb(n,k)


def binomial_distances(n):
    bin0 = binomial(n)
    M = distances(len(bin0),0.5)
    xx = []
    yy = []
    for k in range(1,n):
        print("Computing result ",k,":")
        p = k/n
        print("Finding binomial")
        bin1 = binomial(n,p)
        print("Applying Sinkhorn")
        dist = ot_sinkhorn_value(M,bin0,bin1)
        xx.append(p)
        yy.append(dist)
    plt.plot(xx,yy)
    plt.show()

# binomial_distances(100)

def Gaussian(t,t0,sigma):
    return np.exp( -(t-t0)**2/(2*sigma**2) )

def normalize(p):
    return p/sum(p)

def show_transport(n,lbd=1):
    t = np.arange(n)
    r0 = normalize(Gaussian(t,n*0.1,n/20)*0.5+Gaussian(t,n*0.5,n/10)+0.02)
    r1 = normalize(0.5*Gaussian(t,n*0.8,n/20)+Gaussian(t,n*0.5,n/30)+0.02)
    P = ot_sinkhorn(distances(len(r0),p=2),r0,r1,lbd,1e-3,10000)
    ax = plt.subplot2grid((3,3),(1,1),rowspan=2,colspan=2)
    ax.imshow(np.log(P+1e-5),interpolation='bicubic')
    ax.set_aspect('auto')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax = plt.subplot2grid((3,3),(1,0),rowspan=2)
    ax.plot(r0,t)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax = plt.subplot2grid((3,3),(0,1),colspan=2)
    ax.plot(t,r1)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.show()

def show_error(n=200,lbds=[0.1,0.5,1,5,10,1/.06]):
    t = np.arange(n)
    r0 = normalize(Gaussian(t,n*0.1,n/20)*0.5+Gaussian(t,n*0.5,n/10)+0.02)
    r1 = normalize(0.5*Gaussian(t,n*0.8,n/20)+Gaussian(t,n*0.5,n/30)+0.02)
    ax = plt.axes()
    for lbd in lbds:
        P,err_r = ot_sinkhorn_map_error(distances(len(r0),p=2),r0,r1,lbd=lbd,epsilon=-1,iter_bound=5000)
        lbl = "eps = %.2f" % (1/lbd)
        ax.semilogy(err_r,label=lbl)
    ax.legend()
    ax.set_xlabel("itération")
    ax.set_ylabel("erreur")
    plt.show()


show_transport(200,10)
#show_error()
