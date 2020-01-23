import numpy as np 
import matplotlib.pyplot as plt 

savePlots = 1
showPlots = 1


N = 100 # Note: N = (N+1) from the slides
k = np.arange(1,N)

omega = 1.0
lam = 1.0 - 2.0*omega*np.sin(k*np.pi/(2.0*N))*np.sin(k*np.pi/(2.0*N))



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')

plt.title('Error mode damping for Jacobi')
plt.xlabel('$k$')
plt.ylabel(r'$\tilde\lambda$')
plt.plot(k, np.zeros(N-1),':')
p1 = plt.plot(k, lam)
# plt.legend([p1],['Jacobi damping factor'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 

if savePlots:
   filename = 'jacobiModeDamping.png'
   plt.savefig(filename, bbox_inches="tight")


plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')

plt.title('Error mode damping for weighted Jacobi')
plt.xlabel('$k$')
plt.ylabel(r'$\tilde\lambda$')
j = 1
for omega in [1.0, 2.0/3.0, 1.0/2.0, 1.0/3.0]:
   lam = 1.0 - 2.0*omega*np.sin(k*np.pi/(2.0*N))*np.sin(k*np.pi/(2.0*N))
   plt.plot(k, lam, color=colors[j])
   j = j+1
plt.plot(k, np.zeros(N-1),':',color=colors[0])
plt.legend(['$\omega = 1$','$\omega = 2/3$','$\omega = 1/2$','$\omega = 1/3$'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 

if savePlots:
   filename = 'weightedJacobiModeDamping.png'
   plt.savefig(filename, bbox_inches="tight")


######### Show the plots #########

if showPlots:
   plt.show()
