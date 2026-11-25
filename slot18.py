import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import brentq, minimize_scalar
lam,d,a,N=.031,.032,.023,6
lg=lam/np.sqrt(1-(lam/(2*a))**2)
k=2*np.pi/lam
xi=2*np.pi*d/lg
def E(t):
    return abs(np.mean(np.exp(1j*np.arange(N)*np.expand_dims(k*d*np.sin(t)-xi,-1)),axis=-1))
def H(t):
    c=np.cos(t)
    return abs(np.divide(np.cos(np.pi/2*np.sin(t)),c,out=np.zeros_like(c),where=abs(c)>1e-10))
if __name__=='__main__':
    print('lambda_g, k, xi',lg,k,xi)
    t=np.linspace(-np.pi/2,np.pi/2,36001)
    for name,f in [('E',E),('H',H)]:
        y=f(t)
        p=np.array([minimize_scalar(lambda x:-f(x),bounds=(t[i-1],t[i+1]),method='bounded').x for i in find_peaks(y)[0]])
        z=np.array([minimize_scalar(f,bounds=(t[i-1],t[i+1]),method='bounded').x for i in find_peaks(-y)[0]]) if name=='E' else np.array([-np.pi/2,np.pi/2])
        h=np.array([brentq(lambda x:f(x)-1/np.sqrt(2),t[i],t[i+1]) for i in np.where((y[:-1]-1/np.sqrt(2))*(y[1:]-1/np.sqrt(2))<0)[0]])
        print(name,'peaks',np.degrees(p),f(p),'zeros',np.degrees(z),'half',np.degrees(h))
        for polar in (False,True):
            fig,ax=plt.subplots(figsize=(7,3.8),subplot_kw={'projection':'polar'} if polar else {})
            cv=lambda x:x if polar else np.degrees(x)
            ax.plot(cv(t),y,color='black')
            ax.scatter(cv(p),f(p),color='black',s=18)
            ax.scatter(cv(z),np.zeros(len(z)),marker='x',color='black')
            ax.scatter(cv(h),np.full(len(h),1/np.sqrt(2)),marker='s',color='black',s=15)
            ax.plot(cv(t),np.full(len(t),1/np.sqrt(2)),':',color='gray')
            ax.set_ylim(0,1.05); ax.grid(True)
            if polar:
                ax.set_theta_zero_location('N'); ax.set_thetamin(-90); ax.set_thetamax(90)
            else: ax.set(xlabel='θ, градуси',ylabel='Нормована ДС',xlim=(-90,90))
            fig.tight_layout()
            fig.savefig(f'slot_{name}_'+('polar' if polar else 'linear')+'.png',dpi=160)
            plt.close(fig)
