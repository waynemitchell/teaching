import numpy as np 
import matplotlib.pyplot as plt 

savePlots = 1
showPlots = 1


N = 128
j = np.arange(1,N)



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')

plt.title('Eigenvectors')
plt.xlabel('$j$')
plt.ylabel('$v_j$')

for k in [1,2,5]:
   v = np.sin(k*j*np.pi/N)
   plt.plot(j, v)

plt.legend(['$k = 1$','$k = 2$','$k = 5$'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

if savePlots:
   filename = 'eigenmodes.png'
   plt.savefig(filename, bbox_inches="tight")

######### Show the plots #########

if showPlots:
   plt.show()
