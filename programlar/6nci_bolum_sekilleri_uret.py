from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]/'figurler';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,'axes.titlesize':10,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','lines.linewidth':1.6,'savefig.bbox':'tight'})
B='#174A72';R='#A84435';G='#666666'
def save(f,n):
 for ext in ['pdf','svg']:f.savefig(OUT/(n+'.'+ext),pad_inches=.08)
 plt.close(f)
f,axs=plt.subplots(1,2,figsize=(6.6,2.8));a=axs[0];a.axis('off');a.set_aspect('equal');a.set(xlim=(-.1,1.25),ylim=(-.15,.85));c0=np.sqrt(3)/2;c1=.5
for pt,label,col in [((1.12,0),r'$|0\rangle$',G),((0,.77),r'$|1\rangle$',G),((c0,c1),r'$|\psi\rangle$',B)]:
 a.annotate('',pt,(0,0),arrowprops={'arrowstyle':'->','lw':1.7,'color':col});a.text(pt[0]+.04,pt[1]+.02,label,fontsize=10)
a.plot([c0,c0],[0,c1],'--',color=G,lw=.8);a.annotate('',(c0,0),(0,0),arrowprops={'arrowstyle':'->','lw':3,'color':R});a.text(.32,-.14,r'$\hat P_0|\psi\rangle=c_0|0\rangle$',fontsize=9,color=R);a.set_title('(a) İzdüşüm bir vektördür')
a=axs[1];a.bar([0,1],[.75,.25],color=[R,B],width=.48);a.set(xticks=[0,1],xticklabels=[r'$|0\rangle$',r'$|1\rangle$'],ylim=(0,1),ylabel='Ölçüm olasılığı');a.text(0,.78,r'$|c_0|^2=3/4$',ha='center',fontsize=9);a.text(1,.28,r'$|c_1|^2=1/4$',ha='center',fontsize=9);a.set_title('(b) Olasılık bir sayıdır');f.tight_layout();save(f,'06_izdusum')
f,a=plt.subplots(figsize=(6.6,3.6));a.set(xlim=(-1,6),ylim=(-.6,6.35));a.set_xticks([]);a.set_ylabel(r'$E/(\hbar\omega)$');a.spines['bottom'].set_visible(False)
for n in range(6):
 y=n+.5;a.hlines(y,0,4.6,color=B,lw=1.2);a.text(4.8,y,rf'$|{n}\rangle$',va='center');a.text(.1,y+.12,rf'$E_{n}=({n}+1/2)\hbar\omega$',fontsize=8,color=G)
a.annotate('',(2.1,2.5),(2.1,3.5),arrowprops={'arrowstyle':'->','lw':2,'color':B});a.text(1.1,3,r'$\hat A\!: \sqrt{3}$',color=B,fontsize=11)
a.annotate('',(3.7,4.5),(3.7,3.5),arrowprops={'arrowstyle':'->','lw':2,'color':R});a.text(3.85,4,r'$\hat A^\dagger\!: \sqrt{4}$',color=R,fontsize=11)
a.text(.4,-.32,r'$\hat A|0\rangle=0$  (sıfır vektörü)',fontsize=10);a.text(4,.07,r'$E_0=\hbar\omega/2$',fontsize=10);save(f,'06_merdiven')
f,axs=plt.subplots(1,2,figsize=(6.6,2.9));y=np.linspace(-3.8,3.8,1500);u0=np.pi**(-.25)*np.exp(-y*y/2);u1=np.sqrt(2)*y*u0
for tau,col,ls,lab in [(0,B,'-',r'$\omega t=0$'),(np.pi/2,G,'--',r'$\omega t=\pi/2$'),(np.pi,R,':',r'$\omega t=\pi$')]:
 density=abs((u0+np.exp(-1j*tau)*u1)/np.sqrt(2))**2;axs[0].plot(y,density,color=col,ls=ls,label=lab)
axs[0].set(xlabel=r'$x/\ell$',ylabel=r'$\ell\,|\psi(x,t)|^2$');axs[0].legend(frameon=False,fontsize=8);axs[0].set_title('(a) Schrödinger: yoğunluk değişir')
t=np.linspace(0,2*np.pi,500);psi0=np.array([1,1])/np.sqrt(2);X=np.array([[0,1],[1,0]])/np.sqrt(2);qS=[];qH=[]
for tau in t:
 U=np.diag(np.exp(-1j*np.array([.5,1.5])*tau));psit=U@psi0;qS.append(np.vdot(psit,X@psit).real);qH.append(np.vdot(psi0,U.conj().T@X@U@psi0).real)
axs[1].plot(t,qS,color=B,label='Schrödinger');axs[1].plot(t,qH,color=R,ls='--',label='Heisenberg');axs[1].set(xlabel=r'$\omega t$',ylabel=r'$\langle x\rangle/\ell$',xticks=[0,np.pi,2*np.pi],xticklabels=['0',r'$\pi$',r'$2\pi$']);axs[1].legend(frameon=False,fontsize=8);axs[1].set_title('(b) Beklenen değerler aynıdır');axs[1].axhline(0,color=G,lw=.5);f.tight_layout();save(f,'06_resimler')
print('3 vektörel şekil PDF ve SVG olarak üretildi.')
