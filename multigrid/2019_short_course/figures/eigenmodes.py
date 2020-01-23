import numpy as np 
import matplotlib.pyplot as plt 

savePlots = 1
showPlots = 1


N = 16 # Note: N = (N+1) from the slides
k = 4
j = np.arange(1,N)

v = np.sin(k*j*np.pi/N)



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')

plt.title('Smooth Mode')
plt.xlabel('$j$')
plt.xlim((0,16))
plt.xticks(np.arange(0, 16.1, step=2))
plt.ylabel('$v_j$')
p1 = plt.plot(j, v, '-x')
plt.legend(['$k = 4, N = 15$'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

if savePlots:
   filename = 'smoothMode.png'
   plt.savefig(filename, bbox_inches="tight")

N = 8
k = 4
j = np.arange(1,N)

v = np.sin(k*j*np.pi/N)


p1 = plt.plot(2*j, v, '-or')
plt.legend(['$k = 4, N = 15$','$k = 4, N = 7$'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

if savePlots:
   filename = 'modeCoarse.png'
   plt.savefig(filename, bbox_inches="tight")



######### Show the plots #########

if showPlots:
   plt.show()
