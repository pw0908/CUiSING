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
        np.random.seed()
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

class Ising2DVect(object):
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
        # return -1*J*np.sum(A*np.add.reduce([np.roll(A,1,axis=0),np.roll(A,1,axis=1)])) - h*N*m
        return -1*J*np.sum(A*self.diff_neigh())/2.0 - h*N*m

    def calcDeltaHamiltonian(self,i,j):
        J = self.J
        h = self.h
        A = self.A
        n = self.n
        # return 2.0*(J*A[i,j]*(A[i,np.mod((n+np.mod(j+1,n)),n)]+A[i,np.mod((n+np.mod(j-1,n)),n)]+
        #                   A[np.mod((n+np.mod(i+1,n)),n),j]+A[np.mod((n+np.mod(i-1,n)),n),j])+
        #                   A[i,j]*h)
        return 2.0*(J*A[i,j]*(A[i,(n+((j+1)%n))%n]+A[i,(n+((j-1)%n))%n]+
                          A[(n+((i+1)%n))%n,j]+A[(n+((i-1)%n))%n,j])+
                          A[i,j]*h)
    
    def diff_neigh(self):
        A = self.A
        n = self.n
        # dnn_flat = [(A[i,(n+((j+1)%n))%n]+A[i,(n+((j-1)%n))%n]+
        #                   A[(n+((i+1)%n))%n,j]+A[(n+((i-1)%n))%n,j]) for i,j in pos]
        # dnn = np.reshape(np.array(dnn_flat),(n,n),order='C')
        # return dnn
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
        return self.M, self.E