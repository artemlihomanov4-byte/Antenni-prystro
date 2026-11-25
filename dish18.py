import numpy as np
from scipy.special import jv
from scipy.optimize import brentq, minimize_scalar
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
lam,D,f=.031,1.,.3
k=2*np.pi/lam; R=D/2; p=2*f; v=3.5*R/p
x,w=leggauss(100); x=(x+1)/2; w=w/2
def F(t,plane):
    t=np.asarray(t); u=k*R*np.sin(t); z=u[...,None]*x
    A=np.sum(w*x*jv(0,v*x)*jv(0,z),axis=-1)
    B=np.sum(w*x*jv(0,z),axis=-1)
    C=np.sum(w*x*jv(2,1.5*v*x)*jv(2,z),axis=-1)
    return np.cos(t/2)**2*(.74*A+.26*B+(.25 if plane=='E' else -.25)*C)/(.74*jv(1,v)/v+.13)
def analyze(plane):
    t=np.linspace(0,np.pi/2,5001); y=F(t,plane)
    zeros=[brentq(lambda q:F(q,plane),a,b) for a,b,ya,yb in zip(t[:-1],t[1:],y[:-1],y[1:]) if ya*yb<0]
    half=brentq(lambda q:F(q,plane)-1/np.sqrt(2),0,zeros[0]); peaks=[]
    for a,b in zip(zeros,zeros[1:]+[np.pi/2]):
        q=minimize_scalar(lambda q:-abs(F(q,plane)),bounds=(a,b),method='bounded').x; peaks.append((q,abs(F(q,plane))))
    return half,zeros,peaks
if __name__=='__main__':
    for plane in ['E','H']:
        h,z,s=analyze(plane); print(plane,2*np.degrees(h),np.degrees(z[:4]),[(np.degrees(q),a) for q,a in s[:4]])
        for polar in [False,True]:
            t=np.linspace(-np.pi/2 if polar else 0,np.pi/2,4001); fig,ax=plt.subplots(figsize=(7,3.7),subplot_kw={'projection':'polar'} if polar else {}); cv=lambda t:t if polar else np.degrees(t)
            ax.plot(cv(t),abs(F(t,plane)),color='black'); ax.plot(cv(t),np.full(len(t),1/np.sqrt(2)),':',color='gray')
            for sign in ([-1,1] if polar else [1]):
                ax.scatter(cv(sign*h),1/np.sqrt(2),color='black',s=15); ax.scatter(cv(sign*np.array(z)),np.zeros(len(z)),marker='x',color='black',s=12); ax.scatter(cv(sign*np.array([q for q,a in s])),[a for q,a in s],s=10,color='black')
            if polar: ax.set_theta_zero_location('N'); ax.set_thetamin(-90); ax.set_thetamax(90)
            else: ax.set(xlabel='θ, градуси',ylabel='Нормована ДС',xlim=(0,90))
            ax.set_ylim(0,1.05); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f'dish_{plane}_{"polar" if polar else "linear"}.png',dpi=180); plt.close(fig)
