from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial
from scipy.optimize import brentq
from matplotlib.patches import Rectangle
OUT=Path(__file__).resolve().parents[1]/'figurler'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,'axes.labelsize':11,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','lines.linewidth':1.6,'axes.linewidth':.7,'savefig.bbox':'tight'})
blue='#174A72';red='#A84435';gray='#666666'
def save(fig,name):
 for ext in ['pdf','svg']:fig.savefig(OUT/(name+'.'+ext),pad_inches=.08)
 plt.close(fig)
def axis(ax,xlabel,ylabel):
 ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.axhline(0,color=gray,lw=.7);ax.axvline(0,color=gray,lw=.7)
def delta(ax,where,bottom=-.55):
 ax.annotate('',(where,bottom),(where,0),arrowprops={'arrowstyle':'-|>','color':red,'lw':1.8})
x=np.linspace(-4,4,1601)
f,ax=plt.subplots(figsize=(6.4,2.7));axis(ax,r'$\kappa x$',r'$u(x)/\sqrt{\kappa}$');ax.plot(x,np.exp(-abs(x)),color=blue);delta(ax,0);ax.set_ylim(-.65,1.2);ax.annotate(r'$u^{\prime}(0^-)=+\kappa u(0)$',(-.3,.74),(-3.7,1.08),fontsize=10,arrowprops={'arrowstyle':'-','lw':.6});ax.annotate(r'$u^{\prime}(0^+)=-\kappa u(0)$',(.3,.74),(.8,1.08),fontsize=10,arrowprops={'arrowstyle':'-','lw':.6});ax.text(.15,-.45,r'$V(x)=-g\delta(x)$',color=red);save(f,'delta_tek_bagli')
lam=2.5
yplus=brentq(lambda y:2*y-lam*(1+np.exp(-2*y)),.00001,lam)
yminus=brentq(lambda y:2*y-lam*(1-np.exp(-2*y)),.00001,lam)
f,axs=plt.subplots(1,2,figsize=(6.5,2.65),sharey=True)
for ax,y,s,title in zip(axs,[yplus,yminus],[1,-1],['(a) Çift parite','(b) Tek parite']):
 z=np.exp(-y*abs(x-1))+s*np.exp(-y*abs(x+1));z/=np.sqrt(2/y*(1+s*(1+2*y)*np.exp(-2*y)))
 axis(ax,r'$x/a$',r'$\sqrt{a}\,u(x)$');ax.plot(x,z,color=blue);delta(ax,-1,-1);delta(ax,1,-1);ax.set_title(title);ax.set_xticks([-3,-1,0,1,3]);ax.set_ylim(-1.15,1.1)
save(f,'delta_cift_parite')
f,axs=plt.subplots(1,2,figsize=(6.5,2.9));y=np.linspace(.002,3,1400)
axs[0].plot(y,np.tanh(y),color=blue,label=r'$\tanh y$');axs[0].plot(y,lam/y-1,color=red,label=r'$\lambda/y-1$');axs[0].plot(yplus,np.tanh(yplus),'o',ms=4,color='black');axs[0].set_title(r'(a) Çift: $\lambda=2.5$')
axs[1].plot(y,np.tanh(y),color=blue,label=r'$\tanh y$')
for l,style in [(.7,':'),(1,'--'),(2.5,'-')]:
 yy=y[y<l];axs[1].plot(yy,yy/(l-yy),ls=style,color=red,label=rf'$\lambda={l:g}$')
axs[1].plot(yminus,np.tanh(yminus),'o',ms=4,color='black');axs[1].set_title(r'(b) Tek: $y/(\lambda-y)$')
for ax in axs:ax.set(xlim=(0,3),ylim=(0,1.35),xlabel=r'$y=\kappa a$');ax.legend(frameon=False,fontsize=8,loc='lower right');ax.grid(alpha=.15)
f.tight_layout();save(f,'delta_transandantal')
f,axs=plt.subplots(1,2,figsize=(6.5,2.6));ax=axs[0];ax.axis('off');ax.set(xlim=(-.4,4.5),ylim=(-.7,1.5));ax.plot([0,0],[-.1,1],color=gray,lw=2);ax.plot([-.2,4.4],[0,0],color=gray,lw=.8)
for yy in np.linspace(0,1,8):ax.plot([-.17,0],[yy-.15,yy],color=gray,lw=.6)
t=np.linspace(0,1,400);ax.plot(.1+2.65*t,.4+.13*np.sin(2*np.pi*9*t),color=blue);ax.add_patch(Rectangle((2.85,.05),.7,.7,facecolor='#E7EDF2',edgecolor=blue));ax.text(3.2,.4,r'$m$',ha='center');ax.annotate('',(2.1,1.1),(3.2,1.1),arrowprops={'arrowstyle':'->','color':red});ax.text(2.25,1.25,r'$F=-Kx$',color=red);ax.plot([2.3,2.3],[-.1,.1],color=gray);ax.text(2.3,-.3,r'$0$',ha='center');ax.annotate('',(3.2,-.3),(2.3,-.3),arrowprops={'arrowstyle':'->'});ax.text(3.35,-.3,r'$x$');ax.set_title('(a) Yay–kütle sistemi')
ax=axs[1];xx=np.linspace(-2.2,2.2,500);axis(ax,r'$y=x/\ell$',r'$V/(\hbar\omega)$');ax.plot(xx,xx**2/2,color=blue);ax.set_title('(b) Harmonik potansiyel');f.tight_layout();save(f,'harmonik_klasik')
f,axs=plt.subplots(1,3,figsize=(6.5,2.6))
for j,ax in enumerate(axs):
 ax.axhline(0,color=gray,lw=.7);ax.axvline(0,color=gray,lw=.7);ax.set(xlim=(-2.5,2.5),ylim=(-1.4,2),xlabel=r'$x/a$');ax.set_yticks([]);ax.set_xticks([-1,0,1]);ax.set_title(['(a) Sonsuz kuyu','(b) Sonlu kuyu','(c) Harmonik kuyu'][j],fontsize=9)
 if j==0:
  ax.plot([-1,-1,1,1],[1.8,-1,-1,1.8],color=gray);xx=np.linspace(-1,1,500);even=np.cos(np.pi*xx/2);odd=np.sin(np.pi*xx)
 elif j==1:
  ax.plot([-2.5,-1,-1,1,1,2.5],[1,1,-1,-1,1,1],color=gray);xx=np.linspace(-2.5,2.5,1000)
  # Exact finite-well eigenstates with rho=2.5; continuous derivatives.
  rho=2.5;ze=brentq(lambda z:z*np.tan(z)-np.sqrt(rho*rho-z*z),.01,np.pi/2-.001);zo=brentq(lambda z:-z/np.tan(z)-np.sqrt(rho*rho-z*z),np.pi/2+.001,rho-.001)
  even=np.where(abs(xx)<=1,np.cos(ze*xx),np.cos(ze)*np.exp(-np.sqrt(rho*rho-ze*ze)*(abs(xx)-1)))
  odd=np.where(abs(xx)<=1,np.sin(zo*xx),np.sign(xx)*np.sin(zo)*np.exp(-np.sqrt(rho*rho-zo*zo)*(abs(xx)-1)))
 else:
  xx=np.linspace(-2.5,2.5,1000);ax.plot(xx,.4*xx*xx-1,color=gray);even=np.exp(-xx*xx/2);odd=np.sqrt(2)*xx*even
 ax.plot(xx,.8*even,color=blue,label='Çift');ax.plot(xx,.8*odd,color=red,ls='--',label='Tek')
axs[0].legend(frameon=False,fontsize=8,loc='upper right');f.tight_layout();save(f,'kuyular_parite')
f,axs=plt.subplots(3,2,figsize=(6.4,6.2),sharex=True,sharey=True);y=np.linspace(-4.2,4.2,1800)
for n,ax in enumerate(axs.flat):
 v=eval_hermite(n,y)*np.exp(-y*y/2)/np.sqrt(2**n*factorial(n)*np.sqrt(np.pi));ax.plot(y,v,color=blue if n%2==0 else red);ax.axhline(0,color=gray,lw=.6);ax.axvline(0,color=gray,lw=.6);ax.set_title(rf'$n={n}$ — '+('çift' if n%2==0 else 'tek'));ax.set_ylim(-.85,.85);ax.set_xticks([-4,-2,0,2,4])
for ax in axs[-1]:ax.set_xlabel(r'$y=x/\ell$')
for ax in axs[:,0]:ax.set_ylabel(r'$\sqrt{\ell}\,u_n(x)$')
f.tight_layout(h_pad=1);save(f,'harmonik_ilk_alti')
f,ax=plt.subplots(figsize=(6.4,3.3));xx=np.linspace(-3.5,3.5,800);ax.plot(xx,xx*xx/2,color=gray,label=r'$V(x)/(\hbar\omega)$')
for n in range(6):
 en=n+.5;turn=np.sqrt(2*en);ax.hlines(en,-turn,turn,color=blue,lw=1.2);ax.plot([-turn,turn],[en,en],'o',color=red,ms=3);ax.text(3.1,en,rf'$n={n}$',fontsize=9,va='center')
ax.annotate('',(0,1.5),(0,.5),arrowprops={'arrowstyle':'<->','color':red});ax.text(.15,.97,r'$\hbar\omega$',color=red);ax.set(xlim=(-3.8,3.8),ylim=(0,6.8),xlabel=r'$x/\ell$',ylabel=r'$E/(\hbar\omega)$');ax.legend(frameon=False,loc='upper center');save(f,'harmonik_enerji')
print('7 şekil: PDF ve SVG üretildi.')
