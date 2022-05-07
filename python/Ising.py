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
        self.A = 1 - 2 * np.random.randint(2, size=(n,n))
    def calcHamiltonian(self):
        E = 0.
        J = self.J
        h = self.h
        A = self.A
        n = self.n
        for i in range(n):
            for j in range(n):
                E -=J*A[i,j]*(A[i,np.mod(n+np.mod(j+1,n),n)]+A[i,np.mod(n+np.mod(j-1,n),n)]+
                       A[np.mod(n+np.mod(i+1,n),n),j]+A[np.mod(n+np.mod(i-1,n),n),j])/2.0
                E -= h*A[i,j]
        return E
    def calcDeltaHamiltonian(self,i,j):
        J = self.J
        h = self.h
        A = self.A
        n = self.n
        return 2.0*(J*A[i,j]*(A[i,np.mod((n+np.mod(j+1,n)),n)]+A[i,np.mod((n+np.mod(j-1,n)),n)]+
                          A[np.mod((n+np.mod(i+1,n)),n),j]+A[np.mod((n+np.mod(i-1,n)),n),j])+
                          A[i,j]*h)
    def run(self):
        pos = np.argwhere(self.A)

        self.M[0] = np.sum(self.A)/self.N
        self.E[0] = self.calcHamiltonian()
        for i in range(1,self.n_iters):
            for j in range(self.N):
                dE = self.calcDeltaHamiltonian(pos[j,0], pos[j,1])
                if dE <= 0 or np.random.rand(1) <= np.exp(-dE):
                    self.A[pos[j, 0], pos[j, 1]] *= -1
            self.M[i] = np.sum(self.A) / self.N
            self.E[i] = self.calcHamiltonian()
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
        self.A = 1 - 2 * np.random.randint(2, size=(n,n,n))
    def calcHamiltonian(self):
        E = 0.
        J = self.J
        h = self.h
        A = self.A
        n = self.n
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    E -= J*A[i,j,k]*(A[i,np.mod(n+np.mod(j+1,n),n),k]+A[i,np.mod(n+np.mod(j-1,n),n),k]+
                         A[np.mod(n+np.mod(i+1,n),n),j,k]+A[np.mod(n+np.mod(i-1,n),n),j,k]+
                         A[i,j,np.mod(n+np.mod(k+1,n),n)]+A[i,j,np.mod(n+np.mod(k-1,n),n)])/2.0
                    E -= h*A[i,j,k]
        return E
    def calcDeltaHamiltonian(self,i,j,k):
        J = self.J
        h = self.h
        A = self.A
        n = self.n
        return 2.0*(J*A[i,j,k]*(A[i,np.mod((n+np.mod(j+1,n)),n),k]+A[i,np.mod((n+np.mod(j-1,n)),n),k]+
                            A[np.mod((n+np.mod(i+1,n)),n),j,k]+A[np.mod((n+np.mod(i-1,n)),n),j,k]+
                            A[i,j,np.mod((n+np.mod(k+1,n)),n)]+A[i,j,np.mod((n+np.mod(k-1,n)),n)])+
                            A[i,j,k]*h)
    def run(self):
        pos = np.argwhere(self.A)

        self.M[0] = np.sum(self.A)/self.N
        self.E[0] = self.calcHamiltonian()
        for i in range(1,self.n_iters):
            for j in range(self.N):
                dE = self.calcDeltaHamiltonian(pos[j,0], pos[j,1], pos[j,2])
                if dE <= 0 or np.random.rand(1) <= np.exp(-dE):
                    self.A[pos[j, 0], pos[j, 1], pos[j,2]] *= -1
            self.M[i] = np.sum(self.A) / self.N
            self.E[i] = self.calcHamiltonian()
        return self.M, self.E