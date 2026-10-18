from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.optimize import brentq
from scipy.special import eval_hermite,factorial
OUT=Path(__file__).resolve().parents[1]/'figurler';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,'axes.titlesize':10,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','lines.linewidth':1.5,'savefig.bbox':'tight'})
B='#174A72';R='#A84435';G='#666666'
def save(f,n):
 for ext in ['pdf','svg']:f.savefig(OUT/(n+'.'+ext),pad_inches=.09)
 plt.close(f)
# Discrete spectra in the two confining potentials, mixed spectrum in a finite well.
f,axs=plt.subplots(1,3,figsize=(6.7,3.1))
a=axs[0];a.plot([-.1,0,0,1,1,1.1],[3.4,3.4,0,0,3.4,3.4],color=G);xx=np.linspace(0,1,600)
for n,en in [(1,.6),(2,2.4)]:
 a.hlines(en,0,1,color=G,lw=.6);a.plot(xx,en+.25*np.sin(n*np.pi*xx),color=B);a.text(.04,en+.33,rf'$E_{n}$',fontsize=9)
a.set(xlim=(-.15,1.15),ylim=(-.15,3.65),xlabel=r'$x/a$');a.set_title('(a) Sonsuz kuyu');a.set_yticks([])
a=axs[1];xx=np.linspace(-2.2,2.2,600);a.plot(xx,xx*xx/2,color=G)
for n in range(2):
 en=n+.5;v=eval_hermite(n,xx)*np.exp(-xx**2/2)/np.sqrt(2**n*factorial(n)*np.sqrt(np.pi));a.hlines(en,-2,2,color=G,lw=.6);a.plot(xx,en+.34*v,color=B);a.text(-2,en+.13,rf'$E_{n}$',fontsize=9)
a.set(xlim=(-2.2,2.2),ylim=(-.15,2.65),xlabel=r'$x/\ell$');a.set_title('(b) Harmonik salınıcı');a.set_yticks([])
a=axs[2];xx=np.linspace(-2.7,2.7,1200);a.axhspan(0,1.4,color='#E8EEF3',zorder=-1);a.plot([-2.7,-1,-1,1,1,2.7],[0,0,-2,-2,0,0],color=G)
rho=2.5
for n,par in enumerate([1,-1]):
 z=brentq(lambda z:z*np.tan(z)-np.sqrt(rho*rho-z*z),.01,np.pi/2-.001) if par==1 else brentq(lambda z:-z/np.tan(z)-np.sqrt(rho*rho-z*z),np.pi/2+.001,rho-.001)
 kap=np.sqrt(rho*rho-z*z);en=-2*(1-z*z/rho**2)
 v=np.where(abs(xx)<=1,np.cos(z*xx),np.cos(z)*np.exp(-kap*(abs(xx)-1))) if par==1 else np.where(abs(xx)<=1,np.sin(z*xx),np.sign(xx)*np.sin(z)*np.exp(-kap*(abs(xx)-1)))
 a.hlines(en,-2.7,2.7,color=G,lw=.6);a.plot(xx,en+.22*v,color=B);a.text(-2.6,en+.15,rf'$E_{n+1}$',fontsize=8)
ecoef=2/rho**2;energy=.7;ko=np.sqrt(energy/ecoef);qi=np.sqrt((energy+2)/ecoef)
vsc=np.where(abs(xx)<=1,np.cos(qi*xx),np.cos(qi)*np.cos(ko*(abs(xx)-1))-qi/ko*np.sin(qi)*np.sin(ko*(abs(xx)-1)))
a.plot(xx,energy+.18*vsc/np.max(abs(vsc)),color=R);a.text(-2.5,1.13,r'$E>0$: sürekli',fontsize=8);a.text(-2.5,-2.45,r'$E<0$: kesikli',fontsize=8);a.set(xlim=(-2.7,2.7),ylim=(-2.7,1.5),xlabel=r'$x/a$');a.set_title('(c) Sonlu kuyu');a.set_yticks([0]);f.tight_layout(w_pad=.9);save(f,'05_spektrumlar')
# The vector analogy: coefficients as projections, then the resolution kernel.
f,axs=plt.subplots(1,2,figsize=(6.7,2.9));a=axs[0];a.set_aspect('equal');a.set(xlim=(-.2,2.6),ylim=(-.2,2.2));a.axis('off')
for end,label in [((2.5,0),r'$\mathbf{e}_1$'),((0,2),r'$\mathbf{e}_2$'),((1.8,1.4),r'$\mathbf{v}$')]:
 a.annotate('',end,(0,0),arrowprops={'arrowstyle':'->','color':B if label==r'$\mathbf{v}$' else G,'lw':1.8});a.text(end[0]+.07,end[1]+.03,label)
a.plot([1.8,1.8],[0,1.4],'--',color=R,lw=1);a.plot([0,1.8],[1.4,1.4],'--',color=R,lw=1);a.text(.8,-.2,r'$v_1$',color=R);a.text(-.25,.7,r'$v_2$',color=R);a.set_title('(a) Bileşenlere ayırma')
a=axs[1];x=np.linspace(0,1,2000);y=.4
for N,ls,col in [(6,'--',G),(20,'-',B)]:
 K=2*np.sum(np.sin(np.arange(1,N+1)[:,None]*np.pi*x)*np.sin(np.arange(1,N+1)[:,None]*np.pi*y),axis=0);a.plot(x,K,label=rf'$N={N}$',ls=ls,color=col,lw=1.2)
a.axvline(y,ls=':',color=R);a.set(xlabel=r'$x/a$',ylabel=r'$aK_N(x,y)$');a.set_title(r'(b) $y/a=0.4$: sonlu toplam');a.legend(frameon=False,fontsize=8);f.tight_layout();save(f,'05_tamlik')
# Free-particle degenerate eigenspaces.
f,axs=plt.subplots(1,2,figsize=(6.7,2.8));x=np.linspace(-np.pi,np.pi,1200)
for a in axs:a.axhline(0,color=G,lw=.6);a.axvline(0,color=G,lw=.6);a.set(xlabel=r'$kx$',ylim=(-1.25,1.4),xticks=[-np.pi,0,np.pi],xticklabels=[r'$-\pi$','0',r'$\pi$'])
axs[0].plot(x,np.cos(x),color=B,label=r'$c(x)=\cos kx$');axs[0].plot(x,np.sin(x),color=R,ls='--',label=r'$s(x)=\sin kx$');axs[0].set_title('(a) Enerji ve parite bazı');axs[0].legend(frameon=False,fontsize=8,loc='upper right')
axs[1].plot(x,np.cos(x),color=B,label=r'$\mathrm{Re}\,v_\pm=\cos kx$');axs[1].plot(x,np.sin(x),color=R,ls='--',label=r'$\mathrm{Im}\,v_+=\sin kx$');axs[1].plot(x,-np.sin(x),color=G,ls=':',label=r'$\mathrm{Im}\,v_-=-\sin kx$');axs[1].set_title(r'(b) Ortak özfonksiyonlar: $v_\pm=e^{\pm ikx}$');axs[1].legend(frameon=False,fontsize=8,loc='upper right');f.tight_layout();save(f,'05_dejenere_bazlar')
# Mean force and force at the mean: illustrative density and nonlinear force.
f,axs=plt.subplots(1,2,figsize=(6.7,2.7));x=np.linspace(-2,4,1200);mu=1
for sig,col,ls in [(.25,B,'-'),(.8,R,'--')]:axs[0].plot(x,np.exp(-(x-mu)**2/(2*sig*sig))/(np.sqrt(2*np.pi)*sig),color=col,ls=ls,label=rf'$\Delta x={sig}$')
axs[0].axvline(mu,color=G,lw=.8,ls=':');axs[0].set(xlabel=r'$x$',ylabel=r'$|\psi(x)|^2$');axs[0].set_title(r'(a) Aynı merkez: $\langle x\rangle=1$');axs[0].legend(frameon=False,fontsize=8)
sig=np.linspace(0,1,300);axs[1].plot(sig,-3*(1+sig**2),color=B,label=r'$\langle F(x)\rangle=-3(1+\Delta x^2)$');axs[1].axhline(-3,color=R,ls='--',label=r'$F(\langle x\rangle)=-3$');axs[1].set(xlabel=r'$\Delta x$',ylabel='Kuvvet');axs[1].set_title(r'(b) Yerel örnek: $V(x)=x^3$');axs[1].legend(frameon=False,fontsize=8);f.tight_layout();save(f,'05_ehrenfest_kuvvet')
# Handwritten atom/accelerator comparison, redrawn schematically.
f,axs=plt.subplots(1,2,figsize=(6.7,3));a=axs[0];a.set_aspect('equal');a.axis('off');a.set(xlim=(-1.5,1.5),ylim=(-1.4,1.4))
for r,n in [(.6,1),(1.05,2)]:a.add_patch(Circle((0,0),r,fill=False,edgecolor=B,lw=1.5));a.text(r+.08,.1,rf'$n={n}$',fontsize=10)
a.plot(0,0,'o',ms=8,color=R);a.text(-.25,-.27,'çekirdek',fontsize=8);a.set_title('(a) Atom: kesikli bağlı durumlar');a.text(0,-1.35,'Şematik düzey gösterimi',fontsize=8,ha='center')
a=axs[1];a.set_aspect('equal');a.axis('off');a.set(xlim=(-2.1,2.1),ylim=(-1.4,1.4));a.add_patch(Circle((.5,0),.95,facecolor='#EEF2F6',edgecolor=G,lw=.7));a.add_patch(Rectangle((-2,-.18),.9,.36,facecolor='#DEE6ED',edgecolor=B));a.text(-1.55,-.5,'hızlandırıcı',fontsize=8,ha='center');a.annotate('',(-.4,0),(-1.1,0),arrowprops={'arrowstyle':'->','color':B,'lw':1.8})
for sign,col in [(1,B),(-1,R)]:
 t=np.linspace(0,.95,300);xx=-.42+1.5*t;yy=sign*.75*t*t;a.plot(xx,yy,color=col);a.add_patch(Rectangle((1.05,sign*.85-.12),.3,.24,edgecolor=col,facecolor='white'));a.text(1.45,sign*.85,'D',fontsize=9)
a.text(.35,.25,r'$\mathbf{B}\ \otimes$',fontsize=11);a.text(.45,-1.25,r'$\mathbf{F}=q\,\mathbf{v}\times\mathbf{B}$',fontsize=11,ha='center');a.set_title('(b) Demet: yaklaşık klasik yönlendirme');f.tight_layout();save(f,'05_atom_demeti')
print('5 şekil, PDF ve SVG biçiminde üretildi.')
