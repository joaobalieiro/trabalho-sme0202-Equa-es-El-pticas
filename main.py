import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Parâmetros
h = 0.05  # Malha mais fina para melhor precisão
c = 4 * np.pi**2 - 3
e = np.exp(1)

# Malha
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
    return i * Ny + j

# Discretização da EDP (pontos internos)
for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        k = idx(i, j)
        A[k, idx(i+1, j)] = 1 / h**2
        A[k, idx(i-1, j)] = 1 / h**2
        A[k, idx(i, j+1)] = 3 / h**2 - c / (2 * h)
        A[k, idx(i, j-1)] = 3 / h**2 + c / (2 * h)
        A[k, k] = -8 / h**2  # -2/h² -6/h²

# Condições de contorno
# 1. Bordas verticais (Dirichlet)
for j in range(Ny):
    A[idx(0, j), idx(0, j)] = 1
    A[idx(Nx-1, j), idx(Nx-1, j)] = 1

# 2. Borda inferior (Neumann corrigido)
for i in range(Nx):
    k = idx(i, 0)
    A[k, idx(i, 0)] = -3 / (2 * h)
    A[k, idx(i, 1)] = 4 / (2 * h)
    A[k, idx(i, 2)] = -1 / (2 * h)
    b[k] = -e * np.sin(2 * np.pi * x[i])  # Sinal corrigido aqui

# 3. Borda superior (Robin)
for i in range(Nx):
    k = idx(i, Ny-1)
    A[k, idx(i, Ny-1)] = 3 / (2 * h) + 1
    A[k, idx(i, Ny-2)] = -4 / (2 * h)
    A[k, idx(i, Ny-3)] = 1 / (2 * h)

# Resolver
A = A.tocsr()
u_flat = spsolve(A, b)
u_num = u_flat.reshape((Nx, Ny))

# Verificação de pontos
i_x = np.argmin(np.abs(x - 0.6))  # x = 0.6
j_y = np.argmin(np.abs(y - (-1.0)))  # y = -1
print(f"u_num(0.6, -1) = {u_num[i_x, j_y]:.6f}")
print(f"u_analytical(0.6, -1) = {u_analytical[i_x, j_y]:.6f}")

# Plotagem
X, Y = np.meshgrid(x, y, indexing='ij')
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u_num, cmap='viridis')
ax1.set_title('Solução Numérica')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, u_analytical, cmap='viridis')
ax2.set_title('Solução Analítica')

plt.tight_layout()
plt.show()

# Cálculo do erro absoluto
erro_abs = np.abs(u_num - u_analytical)

# Plotagem da solução analítica e do erro
fig = plt.figure(figsize=(18, 6))

# Subplot 1: Solução Analítica
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u_analytical, cmap='viridis')
ax1.set_title('Solução Analítica')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Subplot 2: Erro Absoluto
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, erro_abs, cmap='hot')
ax2.set_title('Erro Absoluto (Numérico vs. Analítico)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Subplot 3: Mapa de Calor do Erro
ax3 = fig.add_subplot(133)
contour = ax3.contourf(X, Y, erro_abs, cmap='hot')
plt.colorbar(contour, label='Erro Absoluto')
ax3.set_title('Distribuição Espacial do Erro')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

plt.tight_layout()
plt.show()

# Métricas de erro
erro_medio = np.mean(erro_abs)
erro_max = np.max(erro_abs)
print(f"Erro médio absoluto: {erro_medio:.6f}")
print(f"Erro máximo absoluto: {erro_max:.6f}")