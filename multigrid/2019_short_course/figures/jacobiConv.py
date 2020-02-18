import numpy as np 
import matplotlib.pyplot as plt 

savePlots = 1
showPlots = 1

N = 16

# Define zero rhs
f = np.zeros(N-1)

# Define random initial guess
u = np.random.rand(N-1)

# Define initial guess as combination of error modes
j = np.arange(1,N)
u = np.zeros(N-1)
for k in [15]:
   u = u + np.sin(k*j*np.pi/N)

u_init = u.copy()

# 1D Model problem
def ModelProblem1D(N):
    A = np.diag(2*np.ones(N-1)) + np.diag(-1*np.ones(N-2),k=-1) + np.diag(-1*np.ones(N-2),k=1) 
    return A*N*N

A = ModelProblem1D(N)

# Define Gauss-Seidel iteration
def Jacobi(A,u,f,numIterations):
   u_next = np.zeros(len(u))
   for k in range(numIterations):
      for i in range(len(u)):
         # Initialize to the right-hand side
         u_next[i] = f[i]

         # Subtract off (L+U)u
         for j in range(len(u)):
            if j == i:
               diag = A[i,j]
            else:
               u_next[i] = u_next[i] - A[i,j]*u[j]

         # Divide by the diagonal        
         u_next[i] = u_next[i]/diag
      u = u_next.copy()
   return  u

numIterations = 1
u = Jacobi(A,u,f,numIterations)

plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.plot(u_init)
plt.ylim((-1,1))
plt.title('Initial Error')
if savePlots:
   filename = 'jacobiInitialErrHighK.png'
   plt.savefig(filename, bbox_inches="tight")

plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.plot(u)
plt.ylim((-1,1))
plt.title('Remaining Error')
if savePlots:
   filename = 'jacobiRemainingErrHighK.png'
   plt.savefig(filename, bbox_inches="tight")

######### Show the plots #########

if showPlots:
   plt.show()
