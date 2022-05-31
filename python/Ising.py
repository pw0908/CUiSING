import numpy as np

class Ising2D(object):
    def __init__(self, n, J, h, n_iters):
        self.n = n
        self.N = n**2
        self.J = J
        self.h = h
        self.n_iters = n_iters
        self.E = np.zeros(n_iters)
        self.M = np.zeros(n_iters)
        self.m = 0.0
        np.random.seed()
        self.A = 1 - 2 * np.random.randint(2, size=(n,n))
    
    def calcHamiltonian(self):
        J = self.J
        h = self.h
        A = self.A
        N = self.N
        m = self.m
        return (-1*J*np.sum(A*self.diff_neigh())/2.0 - h*N*m)/(N*(2*J+abs(h)))

    def calcDeltaHamiltonian(self,i,j):
        J = self.J
        h = self.h
        A = self.A
        n = self.n
        return 2.0*(J*A[i,j]*(A[i,(n+((j+1)%n))%n]+A[i,(n+((j-1)%n))%n]+
                          A[(n+((i+1)%n))%n,j]+A[(n+((i-1)%n))%n,j])+
                          A[i,j]*h)
    
    def diff_neigh(self):
        A = self.A
        n = self.n
        return np.add.reduce([np.roll(A,1,axis=0), np.roll(A,-1,axis=0), np.roll(A,1,axis=1), np.roll(A,-1,axis=1)])
    
    def run(self):
        self.m = np.sum(self.A)/self.N
        self.M[0] = self.m
        self.E[0] = self.calcHamiltonian()
        for i in range(1,self.n_iters):
            for j in range(self.n):
                for k in range(self.n):
                    dE = self.calcDeltaHamiltonian(j, k)
                    if dE <= 0 or np.random.rand(1) <= np.exp(-dE):
                        self.A[j, k] *= -1
            self.m = np.sum(self.A)/self.N
            self.M[i] = self.m
            self.E[i] = self.calcHamiltonian()
        with open("output/python_cpu_output.dat","w") as f:
            for i in range(0,self.n_iters):
                if i%10 == 0:
                    f.write(f"{i} {self.E[i]:.5f} {self.M[i]:.5f}\n")
        return self.M, self.E

class Ising3D(object):
    def __init__(self, n, J, h, n_iters):
        self.n = n
        self.N = n**3
        self.J = J
        self.h = h
        self.n_iters = n_iters
        self.E = np.zeros(n_iters)
        self.M = np.zeros(n_iters)
        self.m = 0.0
        np.random.seed()
        self.A = 1 - 2 * np.random.randint(2, size=(n,n,n))
    
    def calcHamiltonian(self):
        J = self.J
        h = self.h
        A = self.A
        N = self.N
        m = self.m
        return (-1*J*np.sum(A*self.diff_neigh())/2.0 - h*N*m)/(N*(3*J+abs(h)))

    def calcDeltaHamiltonian(self,i,j,k):
        J = self.J
        h = self.h
        A = self.A
        n = self.n
        return 2.0*(J*A[i,j,k]*(A[i,(n+((j+1)%n))%n,k]+A[i,(n+((j-1)%n))%n,k]+
                          A[(n+((i+1)%n))%n,j,k]+A[(n+((i-1)%n))%n,j,k]+
                          A[i,j,(n+((k+1)%n))%n]+A[i,j,(n+((k-1)%n))%n])+
                          A[i,j,k]*h)
    
    def diff_neigh(self):
        A = self.A
        dnn = np.add.reduce([np.roll(A,1,axis=0), np.roll(A,-1,axis=0),
                             np.roll(A,1,axis=1), np.roll(A,-1,axis=1),
                             np.roll(A,1,axis=2), np.roll(A,-1,axis=2)])
        return dnn
    
    def run(self):
        self.m = np.sum(self.A)/self.N
        self.M[0] = self.m
        self.E[0] = self.calcHamiltonian()
        for i in range(1,self.n_iters):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        dE = self.calcDeltaHamiltonian(j, k, l)
                        if dE <= 0 or np.random.rand(1) <= np.exp(-dE):
                            self.A[j, k, l] *= -1
            self.m = np.sum(self.A)/self.N
            self.M[i] = self.m
            self.E[i] = self.calcHamiltonian()

        with open("output/python_cpu_output.dat","w") as f:
            for i in range(0,self.n_iters):
                if i%10 == 0:
                    f.write(f"{i} {self.E[i]:.5f} {self.M[i]:.5f}\n")
        return self.M, self.E