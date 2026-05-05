"""
4_esquemas_temporales_latigo.py  –  Comparación de esquemas temporales
=======================================================================
Punto 5 del trabajo  (estabilidad asintótica y esquemas):
  • Comparación Explícito (FTCS/Leapfrog) vs Crank-Nicolson vs Analítica
  • Relación de dispersión numérica: ω_num / ω_analítica  vs  kΔx
  • Amortiguamiento artificial vs dispersión para cada esquema
  • Soluciones estables e inestables para distintos Δt

CC: fijo (x=0) | libre (x=L)
CI: primer modo propio sin(πx/2L)  → solución analítica exacta disponible

Ejecutar:
    python 4_esquemas_temporales_latigo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import solver_explicito, solver_crank_nicolson, impulso_gaussiano, L, C

# ─────────────────────────────────────────────────────────────────
#  Solución analítica exacta (CC fijo-libre, CI = 1er modo propio)
# ─────────────────────────────────────────────────────────────────
def u_analitica(x, t, c=C, Ldom=L):
    lam = np.pi / (2 * Ldom)
    return np.sin(lam * x) * np.cos(c * lam * t)


# ─────────────────────────────────────────────────────────────────
#  Solver Leapfrog con CI analítica (sin impulso en x=0)
# ─────────────────────────────────────────────────────────────────
def solver_leapfrog_analitica(N, c, r, t_max):
    dx = L / (N - 1)
    dt = r * dx / c
    r2 = r ** 2
    steps = int(t_max / dt)

    x     = np.linspace(0, L, N)
    u     = u_analitica(x, 0.0)
    u_old = u_analitica(x, -dt)
    u_new = np.zeros(N)

    stride = max(1, steps // 400)
    frames = [(0.0, u.copy())]

    for s in range(1, steps + 1):
        u_new[1:-1] = 2*u[1:-1] - u_old[1:-1] + r2*(u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0]    = 0.0                                       # Dirichlet
        u_new[-1]   = 2*u[-1] - u_old[-1] + r2*(u[-2] - u[-1])  # Neumann (punto fantasma)
        if s % stride == 0:
            frames.append((s * dt, u_new.copy()))
        u_old[:] = u; u[:] = u_new

    return frames, x, dt


def solver_cn_analitica(N, c, r, t_max):
    from scipy.linalg import solve_banded
    dx = L / (N - 1)
    dt = r * dx / c
    r2 = r ** 2
    al = -r2 / 2
    be = 1.0 + r2
    be_f = 1.0 + r2 / 2

    steps = int(t_max / dt)
    x     = np.linspace(0, L, N)
    u     = u_analitica(x, 0.0)
    u_old = u_analitica(x, -dt)

    M  = N - 1
    ab = np.zeros((3, M))
    ab[1, :]  = be; ab[1, -1]  = be_f
    ab[0, 1:] = al; ab[2, :-1] = al

    stride = max(1, steps // 400)
    frames = [(0.0, u.copy())]
    rhs = np.zeros(M)

    for s in range(1, steps + 1):
        rhs[:-1] = (2*u[1:-1] - u_old[1:-1] + (r2/2)*(u_old[2:] - 2*u_old[1:-1] + u_old[:-2]))
        rhs[-1] = (2*u[-1] - u_old[-1] + (r2/2)*(u_old[-2] - u_old[-1]))

        u_new_inner = solve_banded((1, 1), ab, rhs)
        u_new      = np.empty(N)
        u_new[0]   = 0.0
        u_new[1:]  = u_new_inner

        if s % stride == 0:
            frames.append((s * dt, u_new.copy()))
        u_old[:] = u; u[:] = u_new

    return frames, x, dt


# ─────────────────────────────────────────────────────────────────
#  Parámetros
# ─────────────────────────────────────────────────────────────────
N     = 80
r_vals_compare = [0.5, 0.8, 0.95]
T_COMP = 3.0

print("Calculando esquemas temporales...")

# ─────────────────────────────────────────────────────────────────
#  Figura 1: Comparación esquemas en distintos tiempos
# ─────────────────────────────────────────────────────────────────
r_plot = 0.80
t_targets = [0.5, 1.0, 2.0, 3.0]

fr_lf, x_lf, dt_lf = solver_leapfrog_analitica(N=N, c=C, r=r_plot, t_max=T_COMP)
fr_cn, x_cn, dt_cn = solver_cn_analitica(N=N, c=C, r=r_plot, t_max=T_COMP)

t_lf = np.array([f[0] for f in fr_lf])
t_cn_arr = np.array([f[0] for f in fr_cn])
U_lf = np.array([f[1] for f in fr_lf])
U_cn = np.array([f[1] for f in fr_cn])

x_fine = np.linspace(0, L, 400)

fig1, axes1 = plt.subplots(2, len(t_targets), figsize=(14, 7), sharey='row')
fig1.suptitle(f'Comparación de esquemas temporales  –  r = {r_plot:.2f}\n'
              'CC: fijo (x=0) | libre (x=L)  –  CI: sin(πx/2L)',
              fontsize=11, fontweight='bold')

for col, tt in enumerate(t_targets):
    idx_lf = np.argmin(np.abs(t_lf - tt))
    idx_cn = np.argmin(np.abs(t_cn_arr - tt))
    t_eff  = t_lf[idx_lf]
    u_an   = u_analitica(x_fine, t_eff)
    u_lf   = U_lf[idx_lf]
    u_cn   = U_cn[idx_cn]

    for row, (ax, u_num, lbl, col_num) in enumerate([
        (axes1[0, col], u_lf, 'Leapfrog', '#378ADD'),
        (axes1[1, col], u_cn, 'Crank-Nicolson', '#7F77DD'),
    ]):
        ax.plot(x_fine, u_an, 'k-', lw=2.2, alpha=0.8, label='Analítica')
        ax.plot(x_lf, u_num, '--', color=col_num, lw=2, label=lbl)
        ax.fill_between(x_lf, u_analitica(x_lf, t_eff), u_num, alpha=0.2, color=col_num)
        ax.set_title(f'{lbl}  t={t_eff:.2f}s', fontsize=9, color=col_num, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('u(x,t)', fontsize=8)
        ax.axhline(0, color='gray', lw=0.4); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L); ax.set_ylim(-1.4, 1.4)
        ax.axvline(0, color='crimson', lw=1, ls='--', alpha=0.5)
        ax.axvline(L, color='forestgreen', lw=1, ls='--', alpha=0.5)
        if col == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('comparacion_esquemas.png', dpi=150, bbox_inches='tight')
print("Guardado: comparacion_esquemas.png")
plt.show()


# ─────────────────────────────────────────────────────────────────
#  Figura 2: Error L² vs tiempo para cada esquema y cada r
# ─────────────────────────────────────────────────────────────────
def calcular_error_t(solver_func, N, c, r, t_max):
    frames, x, dt = solver_func(N=N, c=C, r=r, t_max=t_max)
    t_arr = np.array([f[0] for f in frames])
    errs  = []
    for t_i, u_i in frames:
        u_ex = u_analitica(x, t_i)
        errs.append(np.sqrt(np.mean((u_i - u_ex)**2)))
    return t_arr, np.array(errs)

fig2, axes2 = plt.subplots(1, len(r_vals_compare), figsize=(14, 5), sharey=True)
fig2.suptitle('Error L²(t) por esquema para distintos r = cΔt/Δx\n'
              '(N=80, CI analítica, CC fijo-libre)',
              fontsize=11, fontweight='bold')

for ax, rv in zip(axes2, r_vals_compare):
    t_l, e_l = calcular_error_t(solver_leapfrog_analitica, N, C, rv, T_COMP)
    t_c, e_c = calcular_error_t(solver_cn_analitica,       N, C, rv, T_COMP)
    ax.semilogy(t_l, e_l, color='#378ADD', lw=1.8, label='Leapfrog')
    ax.semilogy(t_c, e_c, color='#7F77DD', lw=1.8, ls='--', label='Crank-Nicolson')
    ax.set_title(f'r = {rv:.2f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('t [s]', fontsize=9); ax.set_ylabel('||e(t)||_L²', fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.35, which='both')

plt.tight_layout()
plt.savefig('error_esquemas_vs_t.png', dpi=150, bbox_inches='tight')
print("Guardado: error_esquemas_vs_t.png")
plt.show()


# ─────────────────────────────────────────────────────────────────
#  Figura 3: Relación de dispersión numérica
# ─────────────────────────────────────────────────────────────────
xi_arr  = np.linspace(1e-6, np.pi, 500)   # kΔx ∈ (0, π)

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle('Relación de dispersión numérica\n'
              'ω_num(kΔx) / ω_exacta(kΔx)  vs  kΔx  ∈  (0, π)',
              fontsize=11, fontweight='bold')

# ── Leapfrog / FTCS ──
ax_lf = axes3[0]
ax_lf.axhline(1.0, color='k', lw=1.5, ls=':', label='Relación exacta (= 1)')
colors_r = ['#1D9E75', '#378ADD', '#D85A30', '#9932CC']
for rv, clr in zip([0.3, 0.6, 0.9, 1.0], colors_r):
    mu   = rv * np.sin(xi_arr / 2)
    arg  = np.clip(2 * mu, -1, 1)    # argumento de arcsin
    omega_num = 2 * np.arcsin(rv * np.abs(np.sin(xi_arr / 2))) / (rv * xi_arr)
    # solo para modos que no se vuelven evanescentes
    estable = (rv * np.abs(np.sin(xi_arr / 2)) <= 1)
    om_plot  = np.where(estable, omega_num, np.nan)
    ax_lf.plot(xi_arr / np.pi, om_plot, color=clr, lw=2, label=f'r = {rv}')

ax_lf.set_xlabel('kΔx / π', fontsize=11); ax_lf.set_ylabel('ω_num / ω_exacta', fontsize=11)
ax_lf.set_title('Leapfrog / FTCS explícito', fontsize=10, fontweight='bold')
ax_lf.set_xlim(0, 1); ax_lf.set_ylim(0.8, 1.05)
ax_lf.legend(fontsize=9); ax_lf.grid(True, alpha=0.35)
ax_lf.fill_between([0, 1], [0.99, 0.99], [1.01, 1.01], color='green', alpha=0.08,
                   label='±1% exacto')

# ── Crank-Nicolson ──
ax_cn = axes3[1]
ax_cn.axhline(1.0, color='k', lw=1.5, ls=':', label='Relación exacta (= 1)')
for rv, clr in zip([0.3, 0.6, 0.9, 1.5, 2.5], ['#1D9E75', '#378ADD', '#7F77DD', '#D85A30', '#9932CC']):
    # CN: factor de amplificación exactamente unitario, pero fase errónea
    # Para CN:  tan(ω_num Δt/2) = r·sin(kΔx/2)
    #  → ω_num = (2/Δt) arctan(r sin(kΔx/2))
    xi_h = xi_arr / 2
    arg_cn = rv * np.sin(xi_h)
    om_cn  = (2 * np.arctan(arg_cn)) / (rv * xi_arr)
    ax_cn.plot(xi_arr / np.pi, om_cn, color=clr, lw=2, label=f'r = {rv}')

ax_cn.set_xlabel('kΔx / π', fontsize=11); ax_cn.set_ylabel('ω_num / ω_exacta', fontsize=11)
ax_cn.set_title('Crank-Nicolson (implícito)', fontsize=10, fontweight='bold')
ax_cn.set_xlim(0, 1); ax_cn.set_ylim(0.5, 1.05)
ax_cn.legend(fontsize=9); ax_cn.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig('dispersion_esquemas.png', dpi=150, bbox_inches='tight')
print("Guardado: dispersion_esquemas.png")
plt.show()


# ─────────────────────────────────────────────────────────────────
#  Figura 4: Error de fase y amplitud acumulados en el tiempo
# ─────────────────────────────────────────────────────────────────
T_LONG = 10.0
r_test = 0.80

fr_lf_l, x_l, _ = solver_leapfrog_analitica(N=N, c=C, r=r_test, t_max=T_LONG)
fr_cn_l, x_c, _ = solver_cn_analitica(N=N, c=C, r=r_test, t_max=T_LONG)

# Amplitud del 1er modo (proyección sobre sin(πx/2L))
phi = np.sin(np.pi * x_l / (2 * L))
dx_l = L / (N - 1)

def proyectar_amplitud(frames, phi, dx):
    amps = []
    for t_i, u_i in frames:
        amps.append(2 / L * np.trapezoid(u_i * phi, dx=dx))
    return np.array(amps)

t_lf_l = np.array([f[0] for f in fr_lf_l])
t_cn_l = np.array([f[0] for f in fr_cn_l])
amp_lf = proyectar_amplitud(fr_lf_l, phi, dx_l)
amp_cn = proyectar_amplitud(fr_cn_l, phi, dx_l)
amp_ex = np.cos(np.pi * C * t_lf_l / (2 * L))

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 4.5))
fig4.suptitle(f'Acumulación de error de fase y amplitud  (N={N}, r={r_test:.2f})\n'
              'Proyección sobre el 1er modo propio  sin(πx/2L)',
              fontsize=11, fontweight='bold')

ax_a = axes4[0]
ax_a.plot(t_lf_l, amp_ex,  'k-',  lw=2.2, label='Analítica  cos(ωt)')
ax_a.plot(t_lf_l, amp_lf,  '--',  color='#378ADD', lw=1.8, label='Leapfrog')
ax_a.plot(t_cn_l, amp_cn,  '--',  color='#7F77DD', lw=1.8, label='Crank-Nicolson')
ax_a.set_xlabel('t [s]', fontsize=11); ax_a.set_ylabel('Amplitud modal', fontsize=11)
ax_a.set_title('Amplitud del 1er modo vs tiempo', fontsize=10)
ax_a.legend(fontsize=10); ax_a.grid(True, alpha=0.35)

ax_e = axes4[1]
ax_e.plot(t_lf_l, np.abs(amp_lf - amp_ex), color='#378ADD', lw=1.8, label='Error Leapfrog')
ax_e.plot(t_cn_l, np.abs(amp_cn - amp_ex), color='#7F77DD', lw=1.8, ls='--', label='Error CN')
ax_e.set_xlabel('t [s]', fontsize=11); ax_e.set_ylabel('|error amplitud|', fontsize=11)
ax_e.set_title('Error de amplitud acumulado', fontsize=10)
ax_e.legend(fontsize=10); ax_e.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig('error_fase_amplitud.png', dpi=150, bbox_inches='tight')
print("Guardado: error_fase_amplitud.png")
plt.show()

print("\n=== Resumen de esquemas ===")
for rv in r_vals_compare:
    t_l, e_l = calcular_error_t(solver_leapfrog_analitica, N, C, rv, T_COMP)
    t_c, e_c = calcular_error_t(solver_cn_analitica,       N, C, rv, T_COMP)
    print(f"r={rv:.2f}  →  Leapfrog err_final={e_l[-1]:.3e}  | CN err_final={e_c[-1]:.3e}")
