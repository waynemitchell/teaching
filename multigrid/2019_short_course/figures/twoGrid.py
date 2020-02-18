# Import some python libraries we need
import numpy as np
import matplotlib.pyplot as plt

showPlots = 1
savePlots = 1

# 1D Model problem
def ModelProblem1D(N):
    A = np.diag(2*np.ones(N-1)) + np.diag(-1*np.ones(N-2),k=-1) + np.diag(-1*np.ones(N-2),k=1) 
    return A*N*N

# Define Gauss-Seidel iteration
def GaussSeidel(A,u,f):
    for i in range(len(u)):
        u[i] = (1.0/A[i,i])*(f[i] - np.dot(A[i,:],u) + A[i,i]*u[i])
        
# Linear interpolation in 1D
def LinearInterpolation1D(N):
    N_c = int(N/2)
    P = np.zeros((N-1,N_c-1))
    P[0,0] = 0.5
    P[N-2,N_c-2] = 0.5
    for i in range(2,N-1):
        if i % 2 == 0:
            P[i-1,int((i-1)/2)] = 1
        else:
            P[i-1,int((i-2)/2)] = 0.5
            P[i-1,int((i)/2)] = 0.5
    return P

# Restriction by injection 1D
def RestrictionInjection1D(N):
    N_c = int(N/2)
    R = np.zeros((N_c-1,N-1))
    for i in range(N_c-1):
        R[i,2*i+1] = 1
    return R

# Restriction by full-weighting
def RestrictionFullWeighting1D(N):
    P = LinearInterpolation1D(N)
    R = 0.5*P.transpose()
    return R

# Setup fine-grid, coarse-grid operators, interpolation, restriction
N = 16 # Choose a power of 2
A = ModelProblem1D(N)
P = LinearInterpolation1D(N)
R = RestrictionFullWeighting1D(N)
A_c = ModelProblem1D(int(N/2))

# Solution, initial guess, and RHS
u_final = np.random.rand(N-1)
f = np.dot(A,u_final)
u = np.zeros((N-1,))
u_init = u.copy()

# Two-grid cycle
for it in range(4):
    GaussSeidel(A,u,f) # fine-grid relax
u_post_relax = u.copy()
r = f - np.dot(A,u) # residual calculation
r_c = np.dot(R,r) # residual restriction

# e_c = np.zeros(len(r_c))
# for it in range(4):
#     GaussSeidel(A_c,e_c,r_c)

e_c = np.linalg.solve(A_c, r_c) # coarse-grid solve

e = np.dot(P,e_c) # interpolate correction
u = u + e # add correction
u_corrected = u.copy()

# Plot progress toward solution
plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.title('Effect of relaxation')
plt.plot(u_init,'-o')
plt.plot(u_post_relax,'-o')
plt.plot(u_final,'-o')
plt.legend(['Initial guess','Post relax','Solution'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if savePlots:
   filename = 'solnPostRelax.png'
   plt.savefig(filename, bbox_inches="tight")

# Plot error
plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.title('Error after relaxation')
plt.plot(u_final - u_init,'-o')
plt.plot(u_final - u_post_relax,'-o')
plt.legend(['Initial error','Error post relax'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if savePlots:
   filename = 'errPostRelax.png'
   plt.savefig(filename, bbox_inches="tight")

# Plot progress toward solution
plt.figure()
font = {'size' : 16}
plt.rc('font',**font)
plt.grid(linestyle='dotted')
plt.title('Coarse-grid error correction')
plt.plot(np.arange(len(u_final))/2, u_final - u_post_relax,'-o')
plt.plot(np.arange(len(e_c))+0.5,e_c,'-o')
plt.legend(['Fine-grid error','Coarse-grid approximation'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if savePlots:
   filename = 'coarseGridErr.png'
   plt.savefig(filename, bbox_inches="tight")

# Plot progress toward solution
plt.figure()
plt.title('Solution and iterates')
plt.plot(u_init,'-o')
# plt.plot(u_post_relax,'-o')
plt.plot(u_corrected,'-o')
plt.plot(u_final,'-o')
plt.legend(['Initial guess','Post relax','Coarse correction','Solution'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
if savePlots:
   filename = 'solnCoarseCorrect.png'
   plt.savefig(filename, bbox_inches="tight")


if showPlots:
    plt.show()