import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.optimize import brentq, minimize_scalar
lam,h,d,l=.031,.18,.023,.237
eps,tand=2.2,2e-4
# Fig. 4.7: d/(2*lambda)=0.371; interpolate between eps=2 and 2.5.
# Approximate graph readings Vph/c: 0.85 and 0.75.
xi=1/(.85+.4*(.75-.85))
xi_opt=1+lam/(2*l)
def pattern(t,plane,n):
    x=np.pi*d/lam*np.sin(t)
    aperture=np.divide(2*j1(x),x,out=np.ones_like(x),where=x!=0)
    wave=np.sinc(l/lam*(xi-np.cos(t)))/np.sinc(l/lam*(xi-1))
    f=aperture*wave
    if plane=='E': f=f*np.cos(t)
    if n==2: f=f*np.cos(np.pi*h/lam*np.sin(t))
    return f
def analyze(plane,n):
    f=lambda t:pattern(t,plane,n)
    t=np.linspace(0,np.pi/2,20001)
    y=f(t)
    roots=[brentq(f,a,b) for a,b,ya,yb in zip(t[:-1],t[1:],y[:-1],y[1:]) if ya*yb<0]
    if plane=='E': roots.append(np.pi/2)
    roots=sorted(set(roots))
    half=brentq(lambda x:abs(f(x))-1/np.sqrt(2),0,roots[0])
    peaks=[]
    for a,b in zip(roots,roots[1:]+[np.pi/2]):
        if b-a<1e-10: continue
        p=minimize_scalar(lambda x:-abs(f(x)),bounds=(a,b),method='bounded').x
        peaks.append((p,abs(f(p))))
    print(n,plane,'HPBW:',2*np.degrees(half),'zeros:',np.degrees(roots),'side lobes:',[(round(np.degrees(p),3),round(v,5)) for p,v in peaks])
    return half,roots,peaks
if __name__=='__main__':
    print('xi (graph estimate):',xi,'xi_opt (4.7):',xi_opt)
    for n in (1,2):
        for polar in (False,True):
            fig,ax=plt.subplots(figsize=(8,5),subplot_kw={'projection':'polar'} if polar else {})
            t=np.linspace(-np.pi/2 if polar else 0,np.pi/2,4001)
            for plane,style in [('E','-'),('H','--')]:
                half,roots,peaks=analyze(plane,n)
                coord=lambda x:x if polar else np.degrees(x)
                ax.plot(coord(t),abs(pattern(t,plane,n)),style,label=plane)
                for sign in ([-1,1] if polar else [1]):
                    ax.scatter(coord(sign*half),1/np.sqrt(2),s=20)
                    ax.scatter(coord(sign*np.array(roots)),np.zeros(len(roots)),marker='x')
                    ax.scatter(coord(sign*np.array([p for p,v in peaks])),[v for p,v in peaks],s=12)
            if polar:
                ax.set_theta_zero_location('N');ax.set_thetamin(-90);ax.set_thetamax(90)
            else:
                ax.axhline(1/np.sqrt(2),ls=':',color='gray');ax.set(xlabel='θ, градуси',ylabel='Нормована ДС',xlim=(0,90))
            ax.set_ylim(0,1.05);ax.grid(True);ax.legend();fig.tight_layout()
            fig.savefig(f'rod{n}_'+('polar' if polar else 'linear')+'.png',dpi=180);plt.close(fig)
