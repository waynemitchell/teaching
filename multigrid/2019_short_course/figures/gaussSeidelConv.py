import numpy as np 
import matplotlib.pyplot as plt 

savePlots = 1
showPlots = 1

N = 128

# Define zero rhs
f = np.zeros(N-1)

# Define random initial guess
u = np.random.rand(N-1)

# Define initial guess as combination of error modes
j = np.arange(1,N)
u = np.zeros(N-1)
for k in [1,3,5]:
   u = u + np.sin(k*j*np.pi/N)

u_init = u.copy()

# 1D Model problem
def ModelProblem1D(N):
    A = np.diag(2*np.ones(N-1)) + np.diag(-1*np.ones(N-2),k=-1) + np.diag(-1*np.ones(N-2),k=1) 
    return A*N*N

A = ModelProblem1D(N)

# Define Gauss-Seidel iteration
def GaussSeidel(A,u,f,numIterations):
   for k in range(numIterations):
       for i in range(len(u)):
           # Initialize to the right-hand side
           u[i] = f[i]

           # Subtract off (L+U)u
           for j in range(len(u)):
               if j == i:
                   diag = A[i,j]
               else:
                   u[i] = u[i] - A[i,j]*u[j]

           # Divide by the diagonal        
           u[i] = u[i]/diag
   return  u

numIterations = 10
err = np.zeros(numIterations+1)
err[0] = np.linalg.norm(u)
for it in range(numIterations):
   u = GaussSeidel(A,u,f,1)
   err[it+1] = np.linalg.norm(u)

plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.plot(err/err[0])
plt.xlabel('Iteration')
plt.ylabel('Relative err')
plt.title('Convergence of Gauss-Seidel')
if savePlots:
   filename = 'gaussSeidelConvSmooth.png'
   plt.savefig(filename, bbox_inches="tight")


plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.plot(u_init)
plt.title('Initial Error')
if savePlots:
   filename = 'gaussSeidelInitialErrSmooth.png'
   plt.savefig(filename, bbox_inches="tight")

plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.plot(u)
plt.title('Remaining Error')
if savePlots:
   filename = 'gaussSeidelRemainingErrSmooth.png'
   plt.savefig(filename, bbox_inches="tight")

######### Show the plots #########

if showPlots:
   plt.show()
