import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
lam,a,b=.031,.10,.16
k=2*np.pi/lam
# TE10 aperture, planar phase approximation
def E(t):
    return abs(np.sinc(b*np.sin(t)/lam)*(1+np.cos(t))/2)
def H(t):
    x=a*np.sin(t)/lam
    return abs((np.sinc(x-.5)+np.sinc(x+.5))/(2*np.sinc(.5))*(1+np.cos(t))/2)
def results(fun,zeros):
    half=np.degrees(brentq(lambda t:fun(t)-1/np.sqrt(2),0,zeros[0]))
    peaks=[]
    for l,r in zip(zeros,zeros[1:]+[np.pi/2]):
        p=minimize_scalar(lambda t:-fun(t),bounds=(l,r),method='bounded').x
        peaks.append((np.degrees(p),fun(p)))
    return half, np.degrees(zeros),peaks
ze=[np.arcsin(n*lam/b) for n in range(1,6)]
zh=[np.arcsin(n*lam/a) for n in (1.5,2.5)]
re,rh=results(E,ze),results(H,zh)
if __name__=='__main__':
    print('E',re,'H',rh)
    t=np.linspace(0,np.pi/2,2001)
    fig,ax=plt.subplots(figsize=(8,4.3))
    for fun,res,style,label in [(E,re,'-','E'),(H,rh,'--','H')]:
        ax.plot(np.degrees(t),fun(t),style,color='black',label=label)
        ax.scatter(res[1],np.zeros(len(res[1])),marker='x',color='black')
        ax.scatter(*zip(*res[2]),s=14,color='black')
        ax.plot(res[0],1/np.sqrt(2),'ko',ms=4)
        ax.axvline(res[0],color='gray',ls=':',lw=.8)
    ax.axhline(1/np.sqrt(2),color='gray',ls=':');ax.set(xlabel='θ, градуси',ylabel='Нормована ДС',xlim=(0,90),ylim=(-.035,1.05));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig('horn_linear.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.5),subplot_kw={'projection':'polar'});t=np.linspace(-np.pi/2,np.pi/2,2001)
    for fun,res,style,label in [(E,re,'-','E'),(H,rh,'--','H')]:
        ax.plot(t,fun(t),style,color='black',label=label);ax.scatter(np.radians([-res[0],res[0]]),[.707,.707],s=15,color='black')
    ax.set_theta_zero_location('N');ax.set_thetamin(-90);ax.set_thetamax(90);ax.set_ylim(0,1);ax.legend(loc='lower right');fig.tight_layout();fig.savefig('horn_polar.png',dpi=180)
