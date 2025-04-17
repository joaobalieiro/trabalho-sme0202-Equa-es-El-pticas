import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Parâmetros
h = 0.05
c = 4 * np.pi**2 - 3
e = np.exp(1)

# Malha (x: 0 a 1, y: -1 a 1)
x = np.linspace(0, 1, int(1/h) + 1)
y = np.linspace(-1, 1, int(2/h) + 1)
Nx, Ny = len(x), len(y)

# Solução analítica
u_analytical = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        u_analytical[i, j] = np.exp(-y[j]) * np.sin(2 * np.pi * x[i])

# Sistema linear
N = Nx * Ny
A = lil_matrix((N, N))
b = np.zeros(N)

def idx(i, j):
    """Mapeia (i,j) para índice global, garantindo ordem correta."""
    return i * Ny + j  # i varia primeiro, depois j

# Discretização da EDP (pontos internos)
for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        k = idx(i, j)
        A[k, idx(i+1, j)] = 1 / h**2
        A[k, idx(i-1, j)] = 1 / h**2
        A[k, idx(i, j+1)] = 3 / h**2 - c / (2 * h)  # Termo difusivo + convectivo
        A[k, idx(i, j-1)] = 3 / h**2 + c / (2 * h)
        A[k, k] = -8 / h**2  # -2/h² -6/h²

# Condições de contorno (corrigidas)
# 1. Bordas verticais (Dirichlet)
for j in range(Ny):
    A[idx(0, j), idx(0, j)] = 1  # x=0
    A[idx(Nx-1, j), idx(Nx-1, j)] = 1  # x=1

# 2. Borda inferior (Neumann)
for i in range(Nx):
    k = idx(i, 0)  # y = -1
    A[k, idx(i, 0)] = -3 / (2 * h)
    A[k, idx(i, 1)] = 4 / (2 * h)
    A[k, idx(i, 2)] = -1 / (2 * h)
    b[k] = -e * np.sin(2 * np.pi * x[i])

# 3. Borda superior (Robin)
for i in range(Nx):
    k = idx(i, Ny-1)  # y = 1
    A[k, idx(i, Ny-1)] = 3 / (2 * h) + 1
    A[k, idx(i, Ny-2)] = -4 / (2 * h)
    A[k, idx(i, Ny-3)] = 1 / (2 * h)

# Resolver
A = A.tocsr()
u_flat = spsolve(A, b)
u_num = u_flat.reshape((Nx, Ny))  # Formato (Nx, Ny)

# Plotagem corrigida (sem espelhamento)
X, Y = np.meshgrid(x, y, indexing='ij')  # Garante X[i,j] = x[i], Y[i,j] = y[j]

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u_num, cmap='viridis')
ax1.set_title('Solução Numérica')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, u_analytical, cmap='viridis')
ax2.set_title('Solução Analítica')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.show()

# Verificação robusta de pontos
i_x = np.argmin(np.abs(x - 0.8))  # Índice mais próximo de x=0.6
j_y = np.argmin(np.abs(y - (-1.0)))  # Índice mais próximo de y=-1

print(f"u_num(0.6, -1) = {u_num[i_x, j_y]:.4f}")
print(f"u_analytical(0.6, -1) = {u_analytical[i_x, j_y]:.4f}")
