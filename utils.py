"""
utils.py  –  Ecuación de ondas 1D: Látigo
==========================================
Parámetros, funciones de impulso y solucionadores numéricos
compartidos por todos los scripts del trabajo.

Condiciones de contorno FÍSICAS del látigo:
  • x = 0  (extremo fijo / mango): Dirichlet no homogéneo
            u(0, t) = impulso(t)  → el látigo se agita desde aquí
  • x = L  (extremo libre / punta): Neumann homogéneo
            ∂u/∂x (L, t) = 0     → implementado con punto fantasma

Esquemas implementados:
  - Explícito FTCS / Leapfrog (equivalentes para la ecuación de ondas)
  - Crank-Nicolson (implícito, incondicionalmente estable)
"""

import numpy as np
from scipy.linalg import solve_banded

# ─────────────────────────────────────────────
#  Parámetros físicos
# ─────────────────────────────────────────────
L   = 1.0   # Longitud del látigo [m]
C   = 1.0   # Velocidad de onda   [m/s]  (c² = T/ρ)
RHO = 1.0   # Densidad lineal     [kg/m]

# ─────────────────────────────────────────────
#  Funciones de impulso en el extremo fijo
# ─────────────────────────────────────────────

def impulso_gaussiano(t, A=1.0, t0=0.25, sigma=0.07):
    """Impulso gaussiano suave: pico en t=t0."""
    return A * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))


def impulso_sinusoidal(t, A=1.0, t0=0.0, T_imp=0.5):
    """Medio ciclo senoidal de duración T_imp."""
    if np.isscalar(t):
        return A * np.sin(np.pi * (t - t0) / T_imp) if t0 <= t <= t0 + T_imp else 0.0
    res = np.zeros_like(t, dtype=float)
    mask = (t >= t0) & (t <= t0 + T_imp)
    res[mask] = A * np.sin(np.pi * (t[mask] - t0) / T_imp)
    return res


def impulso_triangular(t, A=1.0, t0=0.1, t1=0.4):
    """Impulso triangular entre t0 y t1, pico en el centro."""
    if np.isscalar(t):
        tc = (t0 + t1) / 2
        if t < t0 or t > t1:
            return 0.0
        return A * (1 - abs(t - tc) / (tc - t0))
    res = np.zeros_like(t, dtype=float)
    tc = (t0 + t1) / 2
    mask = (t >= t0) & (t <= t1)
    res[mask] = A * (1 - np.abs(t[mask] - tc) / (tc - t0))
    return res


# ─────────────────────────────────────────────
#  Solver explícito (Leapfrog)
# ─────────────────────────────────────────────

def solver_explicito(N, c, dt, t_max, func_impulso=None):
    """
    Esquema explícito de segundo orden (Leapfrog):
        u_i^{n+1} = 2 u_i^n - u_i^{n-1} + r²(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
    con r = c·Δt/Δx  (estable si r ≤ 1).

    CC:
      u[0]  = impulso(t+dt)            extremo fijo (Dirichlet)
      u[-1] = u[-2]                    extremo libre (Neumann, punto fantasma)

    Devuelve
    --------
    frames : list of (t, u_array)
    x      : array de nodos espaciales
    r      : número CFL efectivo
    """
    if func_impulso is None:
        func_impulso = impulso_gaussiano

    dx = L / (N - 1)
    r  = c * dt / dx
    r2 = r ** 2

    x     = np.linspace(0, L, N)
    u     = np.zeros(N)          # nivel n
    u_old = np.zeros(N)          # nivel n-1
    u_new = np.zeros(N)          # nivel n+1

    # Condición inicial: reposo total, extremo izquierdo con el impulso en t=0
    u[0]     = func_impulso(0.0)
    u_old[:] = u.copy()

    steps  = int(t_max / dt)
    stride = max(1, steps // 500)   # ~500 fotogramas guardados
    frames = [(0.0, u.copy())]

    for s in range(1, steps + 1):
        t_new = s * dt

        # Nodos interiores
        u_new[1:-1] = (2 * u[1:-1] - u_old[1:-1]
                       + r2 * (u[2:] - 2 * u[1:-1] + u[:-2]))

        # CC izquierda: impulso prescrito
        u_new[0] = func_impulso(t_new)

        # CC derecha: extremo libre — Leapfrog con punto fantasma u_N = u_{N-1}
        u_new[-1] = 2*u[-1] - u_old[-1] + r2 * (u[-2] - u[-1])

        if s % stride == 0:
            frames.append((t_new, u_new.copy()))

        u_old[:] = u
        u[:]     = u_new

    return frames, x, r


# ─────────────────────────────────────────────
#  Solver Crank-Nicolson (implícito)
# ─────────────────────────────────────────────

def solver_crank_nicolson(N, c, dt, t_max, func_impulso=None):
    """
    Esquema Crank-Nicolson para la ecuación de ondas:
        (u^{n+1} - 2u^n + u^{n-1})/dt² = (c²/2)(δ²_x u^{n+1} + δ²_x u^{n-1})

    Sistema tridiagonal en cada paso:
        α u_{i-1}^{n+1} + β u_i^{n+1} + α u_{i+1}^{n+1} = RHS_i

    con α = -r²/2 ,  β = 1 + r²

    CC:
      i = 0   → u^{n+1}_0 = impulso(t+dt)   (Dirichlet, fila eliminada)
      i = N-1 → extremo libre: u_N = u_{N-1} → columna N eliminada,
                coeficiente de u_{N-1} se dobla: β_last = 1 + r²/2

    Devuelve igual que solver_explicito.
    """
    if func_impulso is None:
        func_impulso = impulso_gaussiano

    dx = L / (N - 1)
    r  = c * dt / dx
    al = -r ** 2 / 2        # subdiagonal / superdiagonal
    be =  1.0 + r ** 2      # diagonal principal (nodos interiores)
    be_free = 1.0 + r**2/2  # diagonal en el nodo libre

    x     = np.linspace(0, L, N)
    u     = np.zeros(N)
    u_old = np.zeros(N)

    u[0] = func_impulso(0.0)
    u_old[:] = u.copy()

    # Matriz banda almacenada en formato scipy (ab[0]=super, ab[1]=diag, ab[2]=sub)
    # Trabajamos sobre nodos i = 1 … N-1  (sin el nodo 0, conocido)
    M  = N - 1       # tamaño del sistema lineal
    ab = np.zeros((3, M))
    ab[1, :]   = be              # diagonal
    ab[1, -1]  = be_free         # nodo libre
    ab[0, 1:]  = al              # superdiagonal (ab[0,0] no se usa)
    ab[2, :-1] = al              # subdiagonal   (ab[2,-1] no se usa)

    steps  = int(t_max / dt)
    stride = max(1, steps // 500)
    frames = [(0.0, u.copy())]

    rhs = np.zeros(M)

    for s in range(1, steps + 1):
        t_new = s * dt
        imp   = func_impulso(t_new)

        # RHS para nodos interiores i = 1 … N-2  (índices j = 0 … N-3 en el sistema M)
        # RHS_i = 2 u_i^n - u_i^{n-1}  +  (r²/2) δ²_x u^{n-1}_i
        rhs[:-1] = (2.0 * u[1:-1] - u_old[1:-1]
                    + (r**2 / 2.0) * (u_old[2:] - 2.0*u_old[1:-1] + u_old[:-2]))

        # Corrección por CC izquierda: mover contribución de u^{n+1}_0 (conocido) al RHS
        # Fila j=0 (i=1): al * u^{n+1}_0  se conoce → restarlo de la ecuación
        rhs[0] -= al * imp

        # RHS para el nodo libre i = N-1  (punto fantasma: u_N = u_{N-1})
        # δ²_x u_{N-1}^{n-1} con Neumann: (u_{N-2}^{n-1} - u_{N-1}^{n-1})
        rhs[-1] = (2.0 * u[-1] - u_old[-1]
                   + (r**2 / 2.0) * (u_old[-2] - u_old[-1]))

        u_new_inner = solve_banded((1, 1), ab, rhs)

        u_new      = np.empty(N)
        u_new[0]   = imp
        u_new[1:]  = u_new_inner

        if s % stride == 0:
            frames.append((t_new, u_new.copy()))

        u_old[:] = u
        u[:]     = u_new

    return frames, x, r


# ─────────────────────────────────────────────
#  Energía discreta
# ─────────────────────────────────────────────

def energia_discreta(u_curr, u_prev, dt, dx, c):
    """
    E_k + E_p discretizadas con trapecios.
    u_curr : u^n,  u_prev : u^{n-1}
    """
    vel  = (u_curr - u_prev) / dt
    dudx = np.diff(u_curr) / dx
    Ek   = 0.5 * RHO * dx * np.sum(vel ** 2)
    Ep   = 0.5 * RHO * c**2 * dx * np.sum(dudx ** 2)
    return Ek, Ep
