import numpy as np
import matplotlib.pyplot as plt

# g(x) = (f/T')(T-1(x))

def f(x):
    return np.exp(-(x-5)**2)+2*x*(x<1)+(8<x<9)+0.5*(4<x<4.5)+0.2*(6<x<=8)

def T0(x):
    return 0.27*x**3+0.1*x-0.02*x**4

def T1(x):
    return 10*T0(x)/T0(10)

def T(y):
    x = np.linspace(0,10,10000)
    return np.max(x * (T1(x)<=y))

eps = 1e-1
def T_prime(y):
    return (T(y+eps)-T(y))/eps


def g(y):
    x = T1(y)
    return f(x)/T_prime(x)
    

xx = np.linspace(0,10,300)
fxx = [f(x) for x in xx]
gxx = [g(x) for x in xx]
Txx = [T(x) for x in xx]
plt.plot(xx,fxx,'--',label='$\\mu$')
plt.fill_between(xx,0,fxx)
plt.plot(gxx,xx,'-.',label='$\\nu$')
plt.fill_betweenx(xx,0,gxx)
plt.plot(xx,Txx,'-',label='T')
plt.legend()
plt.show()