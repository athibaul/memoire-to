import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


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
    global b,g,g2
    K = np.size(P,2)
    N = np.size(P,0)
    n = N # width of the convolution kernel
    t = np.linspace(-n/(2*N),n/(2*N),n)
    g = normalize(np.exp(-t**2 / gamma)); g2 =  np.outer(g,g)
    def xi(x):
        return convolve2d(x,g2,'same')

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
    
    return q


def test0():
    n_2 = 20
    r = 16
    N = 2*n_2
    P = np.array([[x*x+y*y < r*r for x in range(-n_2,n_2)] for y in range(-n_2,n_2)])
    plt.subplot(121)
    plt.imshow(P)
    plt.colorbar()
    
    n = 41
    gamma = 0.04**2
    t = np.linspace(-n/(2*N),n/(2*N),n)
    g = normalize(np.exp(-t**2 / gamma)); g2 = np.outer(g,g)
    def xi(x):
        return convolve2d(x,g2,'same')
    
    plt.subplot(122)
    plt.imshow(xi(P))
    plt.colorbar()
    plt.show()

def test1():
    global q
    n = 50
    t = np.arange(n)
    rep1 = normalize(np.array([[(x>3)*(x<10) == (y>n/2) for x in range(n)] for y in range(n)]))
    rep2 = np.outer(Gaussian(t,0.1*n,0.05*n),Gaussian(t,0.5*n,0.2*n))
    rep2 = normalize(rep2)
    
    P = np.zeros((n,n,2))
    P[:,:,0] = rep1; P[:,:,1] = rep2
    number_of_subplots = 5
    coefs = np.linspace(0,1,number_of_subplots)
    for i,alpha in enumerate(coefs):
        lbd = np.array([alpha, 1-alpha])
        ax1 = plt.subplot(1,number_of_subplots,i+1)
        ax1.imshow(barycenter(P,lbd))
    plt.show()



def test2():
    n = 40//2
    r = 4*n//5
    r1 = n//2
    rep1 = [[x*x+y*y < r*r for y in range(-n,n+1)] for x in range(-n,n+1)]
    rep2 = [[ (abs(x-y) < r1 or abs(x+y) < r1) for y in range(-n,n+1)] for x in range(-n,n+1)]
    rep1 = normalize(np.array(rep1))
    rep2 = normalize(np.array(rep2))
    plt.subplot(121);plt.imshow(rep1)
    plt.subplot(122);plt.imshow(rep2)
    plt.show()
    
    n = 2*n+1
    P = np.zeros((n,n,2))
    P[:,:,0] = rep1; P[:,:,1] = rep2
    number_of_subplots = 5
    coefs = np.linspace(0,1,number_of_subplots)
    for i,alpha in enumerate(coefs):
        lbd = np.array([alpha, 1-alpha])
        q = barycenter(P,lbd,gamma=0.05**2)
        ax1 = plt.subplot(1,number_of_subplots,i+1)
        ax1.imshow(q)
    plt.show()
    