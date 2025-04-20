import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ======================================
# PARTE 1: SOLUÇÃO PARA h = 0.05
# ======================================
h = 0.05  # Malha base
c = 4 * np.pi ** 2 - 3
e = np.exp(1)

# Malha
x = np.linspace(0, 1, int(1 / h) + 1)
y = np.linspace(-1, 1, int(2 / h) + 1)
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


# Discretização da EDP
for i in range(1, Nx - 1):
    for j in range(1, Ny - 1):
        k = idx(i, j)
        A[k, idx(i + 1, j)] = 1 / h ** 2
        A[k, idx(i - 1, j)] = 1 / h ** 2
        A[k, idx(i, j + 1)] = 3 / h ** 2 - c / (2 * h)
        A[k, idx(i, j - 1)] = 3 / h ** 2 + c / (2 * h)
        A[k, k] = -8 / h ** 2

# Condições de contorno
for j in range(Ny):
    A[idx(0, j), idx(0, j)] = 1
    A[idx(Nx - 1, j), idx(Nx - 1, j)] = 1

for i in range(Nx):
    k = idx(i, 0)
    A[k, idx(i, 0)] = -3 / (2 * h)
    A[k, idx(i, 1)] = 4 / (2 * h)
    A[k, idx(i, 2)] = -1 / (2 * h)
    b[k] = -e * np.sin(2 * np.pi * x[i])

for i in range(Nx):
    k = idx(i, Ny - 1)
    A[k, idx(i, Ny - 1)] = 3 / (2 * h) + 1
    A[k, idx(i, Ny - 2)] = -4 / (2 * h)
    A[k, idx(i, Ny - 3)] = 1 / (2 * h)

# Resolver
A = A.tocsr()

# ======================================
# GERAR IMAGEM DA ESTRUTURA ESPARSA DA MATRIZ
# ======================================
plt.figure(figsize=(10, 10))
plt.spy(A, markersize=0.05)
plt.title("Estrutura Esparsa da Matriz A (h = 0.05)", fontsize=12)
plt.xlabel("Índice da coluna", fontsize=10)
plt.ylabel("Índice da linha", fontsize=10)
plt.savefig('matriz_esparsa.png', dpi=300, bbox_inches='tight')
plt.close()

u_flat = spsolve(A, b)
u_num = u_flat.reshape((Nx, Ny))

# Verificação de pontos
i_x = np.argmin(np.abs(x - 0.6))
j_y = np.argmin(np.abs(y - (-1.0)))
print(f"u_num(0.6, -1) = {u_num[i_x, j_y]:.6f}")
print(f"u_analytical(0.6, -1) = {u_analytical[i_x, j_y]:.6f}")

# Plotagem das soluções e erro
X, Y = np.meshgrid(x, y, indexing='ij')

# Figura 1: Soluções numérica e analítica
fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u_num, cmap='viridis')
ax1.set_title('Solução Numérica (h=0.05)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig1.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, u_analytical, cmap='viridis')
ax2.set_title('Solução Analítica')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Figura 2: Erro absoluto
erro_abs = np.abs(u_num - u_analytical)
fig2 = plt.figure(figsize=(18, 6))

ax3 = fig2.add_subplot(131, projection='3d')
ax3.plot_surface(X, Y, erro_abs, cmap='hot')
ax3.set_title('Erro Absoluto (3D)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

ax4 = fig2.add_subplot(132)
contour = ax4.contourf(X, Y, erro_abs, levels=50, cmap='hot')
plt.colorbar(contour, label='Erro Absoluto')
ax4.set_title('Distribuição do Erro (2D)')
ax4.set_xlabel('x')
ax4.set_ylabel('y')

# Métricas de erro
erro_medio = np.mean(erro_abs)
erro_max = np.max(erro_abs)
print(f"Erro médio absoluto: {erro_medio:.6f}")
print(f"Erro máximo absoluto: {erro_max:.6f}")

# ======================================
# PARTE 2: ESTUDO DE CONVERGÊNCIA
# ======================================
h_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
erro_max_list = []

for h in h_list:
    x = np.linspace(0, 1, int(1 / h) + 1)
    y = np.linspace(-1, 1, int(2 / h) + 1)
    Nx, Ny = len(x), len(y)

    u_analytical = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            u_analytical[i, j] = np.exp(-y[j]) * np.sin(2 * np.pi * x[i])

    N = Nx * Ny
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(i, j):
        return i * Ny + j

    # Discretização da EDP
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            k = idx(i, j)
            A[k, idx(i+1, j)] = 1 / h**2
            A[k, idx(i-1, j)] = 1 / h**2
            A[k, idx(i, j+1)] = 3 / h**2 - c / (2 * h)
            A[k, idx(i, j-1)] = 3 / h**2 + c / (2 * h)
            A[k, k] = -8 / h**2

    # Condições de contorno
    for j in range(Ny):
        A[idx(0, j), idx(0, j)] = 1
        A[idx(Nx-1, j), idx(Nx-1, j)] = 1

    for i in range(Nx):
        k = idx(i, 0)
        A[k, idx(i, 0)] = -3 / (2 * h)
        A[k, idx(i, 1)] = 4 / (2 * h)
        A[k, idx(i, 2)] = -1 / (2 * h)
        b[k] = -e * np.sin(2 * np.pi * x[i])

    for i in range(Nx):
        k = idx(i, Ny-1)
        A[k, idx(i, Ny-1)] = 3 / (2 * h) + 1
        A[k, idx(i, Ny-2)] = -4 / (2 * h)
        A[k, idx(i, Ny-3)] = 1 / (2 * h)

    A = A.tocsr()
    u_flat = spsolve(A, b)
    u_num = u_flat.reshape((Nx, Ny))

    erro_abs = np.abs(u_num - u_analytical)
    erro_max_list.append(np.max(erro_abs))

# Plotar convergência
h_array = np.array(h_list)
erro_max_array = np.array(erro_max_list)

fig3 = plt.figure(figsize=(8, 6))
plt.loglog(h_array, erro_max_array, 'o-', label='Erro máximo')
plt.loglog(h_array, h_array ** 2, '--', label='Referência O(h²)')
plt.xlabel('Espaçamento da malha (h)')
plt.ylabel('Erro máximo absoluto')
plt.title('Estudo de Convergência')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()