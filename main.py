import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# =============================
# Parâmetros
# =============================
h = 0.05
c = 4 * np.pi**2 - 3
e = np.exp(1)
Lx, Ly = 1.0, 2.0  # comprimento em x, altura em y
Nx = int(1 / h)
Ny = int(2 / h)
x = np.linspace(0, 1, Nx + 1)
y = np.linspace(-1, 1, Ny + 1)
X, Y = np.meshgrid(x, y, indexing='ij')
N = (Nx + 1) * (Ny + 1)

# =============================
# Função de mapeamento 2D -> 1D
# =============================
def idx(i, j):
    return i * (Ny + 1) + j

# =============================
# Matriz A e vetor b
# =============================
A = lil_matrix((N, N))
b = np.zeros(N)

# Montagem do sistema
for i in range(Nx + 1):
    xi = x[i]
    for j in range(Ny + 1):
        k = idx(i, j)

        # Fronteiras verticais: Dirichlet
        if i == 0 or i == Nx:
            A[k, k] = 1
            b[k] = 0

        # Fronteira inferior (Neumann): y = -1 → j = 0
        elif j == 0:
            k1 = idx(i, 1)
            k2 = idx(i, 2)
            A[k, k] = -3 / (2*h)
            A[k, k1] = 4 / (2*h)
            A[k, k2] = -1 / (2*h)
            b[k] = e * np.sin(2 * np.pi * xi)

        # Fronteira superior (Robin): y = 1 → j = Ny
        elif j == Ny:
            k1 = idx(i, Ny - 1)
            k2 = idx(i, Ny - 2)
            A[k, k] = (3 + 2*h) / (2*h)
            A[k, k1] = -4 / (2*h)
            A[k, k2] = 1 / (2*h)
            b[k] = 0

        # Pontos interiores
        else:
            kpx = idx(i + 1, j)
            kmx = idx(i - 1, j)
            kpy = idx(i, j + 1)
            kmy = idx(i, j - 1)

            A[k, kpx] = 1
            A[k, kmx] = 1
            A[k, kpy] = 3 - (c * h) / 2
            A[k, kmy] = 3 + (c * h) / 2
            A[k, k] = -8
            b[k] = 0

# =============================
# Solução do sistema
# =============================
u = spsolve(A.tocsr(), b)
U = u.reshape((Nx + 1, Ny + 1))

# =============================
# Visualização
# =============================
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis')
ax.set_title("Solução Numérica da Equação de Convecção-Difusão Anisotrópica")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
plt.tight_layout()
plt.show()
