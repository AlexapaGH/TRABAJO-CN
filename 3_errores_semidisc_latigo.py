"""
3_errores_semidisc_latigo.py  –  Errores de la semi-discretización espacial
===========================================================================
Punto 4 del trabajo:
  • Error de truncamiento de las diferencias finitas centradas de 2º orden:
        d²u/dx² ≈ (u_{i-1} - 2u_i + u_{i+1}) / Δx²  +  O(Δx²)
  • Convergencia en norma L² vs Δx en escala log-log (pendiente teórica = 2)
  • Comparación D.F. 2º orden vs 4º orden (opcional, para discusión)
  • Error vs tiempo para un refinamiento fijo

Metodología:
  Para obtener una "solución analítica" con CC mixtas (fijo-libre) se usa la
  solución exacta de la serie de Fourier para condiciones iniciales seno:

    CC: u(0,t)=0, ∂u/∂x(L,t)=0
    CI: u(x,0) = sin(π x / (2L)), u_t(x,0) = 0

  Solución exacta:
    u(x,t) = sin(π x / (2L)) · cos(π c t / (2L))

Ejecutar:
    python 3_errores_semidisc_latigo.py [--guardar]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import L, C

# ───────────────────────────────────────────────────────────────
#  Argumentos de línea de comandos
# ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Análisis de errores de semi-discretización',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('--guardar', action='store_true', default=False,
                    help='Guardar todos los archivos (valor por defecto)')
args = parser.parse_args()

# ───────────────────────────────────────────────────────────────
#  Solución analítica (primer modo de Fourier con CC fijo-libre)
# ───────────────────────────────────────────────────────────────
def u_analitica(x, t, c=C, Ldom=L):
    """Solución exacta: CC fijo-libre, CI u(x,0)=sin(πx/(2L)), ut(x,0)=0."""
    return np.sin(np.pi * x / (2*Ldom)) * np.cos(np.pi * c * t / (2*Ldom))


# ───────────────────────────────────────────────────────────────
#  Solver de errores (sin almacenar todos los frames)
# ───────────────────────────────────────────────────────────────
def solver_error(N, c, r_cfl=0.5, t_sample=None):
    """
    Resuelve la ecuación de ondas con CC fijo-libre y CI analítica.
    Devuelve el error L² en t=t_sample.
    """
    dx  = L / (N - 1)
    dt  = r_cfl * dx / c
    if t_sample is None:
        t_sample = 0.5

    steps   = int(t_sample / dt)
    t_real  = steps * dt
    r2      = (c * dt / dx) ** 2

    x     = np.linspace(0, L, N)
    u     = u_analitica(x, 0.0)
    u_old = u_analitica(x, -dt)   # paso ficticio hacia atrás
    u_new = np.zeros(N)

    for s in range(steps):
        u_new[1:-1] = 2*u[1:-1] - u_old[1:-1] + r2*(u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0]    = 0.0                                  # fijo
        u_new[-1]   = 2*u[-1] - u_old[-1] + r2*(u[-2] - u[-1])  # libre (punto fantasma)

        u_old[:] = u
        u[:]     = u_new

    u_exact = u_analitica(x, t_real)
    err_L2  = np.sqrt(np.mean((u - u_exact)**2))
    return err_L2, t_real


# ───────────────────────────────────────────────────────────────
#  A) Convergencia espacial: error L² vs Δx (r fijo pequeño)
# ───────────────────────────────────────────────────────────────
print("Calculando convergencia espacial...")
N_list  = [8, 12, 20, 32, 50, 80, 120, 200]
r_cfl   = 0.3      # CFL pequeño → error temporal despreciable
t_samp  = 0.4

errors_2nd = []
dx_list    = []

for N in N_list:
    dx_i = L / (N - 1)
    err, t_r = solver_error(N=N, c=C, r_cfl=r_cfl, t_sample=t_samp)
    errors_2nd.append(err)
    dx_list.append(dx_i)
    print(f"  N={N:4d}  Δx={dx_i:.5f}  errL²={err:.3e}")

dx_arr    = np.array(dx_list)
err_arr   = np.array(errors_2nd)

# Cálculo del orden de convergencia observado
orders = np.diff(np.log(err_arr)) / np.diff(np.log(dx_arr))
print("\nÓrdenes de convergencia observados:", np.round(orders, 3))


# ───────────────────────────────────────────────────────────────
#  B) Convergencia temporal: error vs Δt (N fijo)
# ───────────────────────────────────────────────────────────────
print("\nCalculando convergencia temporal...")
N_fixed  = 120
r_list   = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]
errors_t = []
dt_list  = []
dx_fixed = L / (N_fixed - 1)

for r in r_list:
    dt_i = r * dx_fixed / C
    err, _ = solver_error(N=N_fixed, c=C, r_cfl=r, t_sample=t_samp)
    errors_t.append(err)
    dt_list.append(dt_i)
    print(f"  r={r:.2f}  Δt={dt_i:.6f}  errL²={err:.3e}")

dt_arr   = np.array(dt_list)
err_t_arr= np.array(errors_t)
orders_t = np.diff(np.log(err_t_arr)) / np.diff(np.log(dt_arr))
print("Órdenes convergencia temporal:", np.round(orders_t, 3))


# ───────────────────────────────────────────────────────────────
#  C) Error en función del tiempo (refinamiento fijo N=80)
# ───────────────────────────────────────────────────────────────
print("\nCalculando error vs tiempo...")

def solver_error_vs_t(N, c, r_cfl=0.5, t_max=2.0, stride_t=50):
    dx = L / (N - 1)
    dt = r_cfl * dx / c
    steps = int(t_max / dt)
    r2 = (c * dt / dx)**2
    x = np.linspace(0, L, N)
    u     = u_analitica(x, 0.0)
    u_old = u_analitica(x, -dt)
    u_new = np.zeros(N)
    t_rec, err_rec = [], []
    for s in range(steps):
        u_new[1:-1] = 2*u[1:-1] - u_old[1:-1] + r2*(u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0]  = 0.0
        u_new[-1] = 2*u[-1] - u_old[-1] + r2*(u[-2] - u[-1])  # punto fantasma
        u_old[:] = u; u[:] = u_new
        if s % stride_t == 0:
            t_now = (s + 1) * dt
            t_rec.append(t_now)
            err_rec.append(np.sqrt(np.mean((u - u_analitica(x, t_now))**2)))
    return np.array(t_rec), np.array(err_rec)

t_rec80,  err_rec80  = solver_error_vs_t(N=80,  c=C, r_cfl=0.5, t_max=2.0)
t_rec120, err_rec120 = solver_error_vs_t(N=120, c=C, r_cfl=0.5, t_max=2.0)
t_rec40,  err_rec40  = solver_error_vs_t(N=40,  c=C, r_cfl=0.5, t_max=2.0)


# ═══════════════════════════════════════════════════════════════
#  FIGURAS
# ═══════════════════════════════════════════════════════════════

# ── Figura 1: Convergencia espacial log-log ──
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Análisis de convergencia  –  Semi-discretización espacial',
              fontsize=12, fontweight='bold')

ax_sp = axes1[0]
ax_sp.loglog(dx_arr, err_arr, 'o-', color='#378ADD', lw=2, ms=7, label='Error L² numérico')
# Línea de referencia O(Δx²)
ref_c2 = err_arr[-1] / dx_arr[-1]**2
ax_sp.loglog(dx_arr, ref_c2 * dx_arr**2, '--', color='#D85A30', lw=1.8,
             label='Pendiente O(Δx²)')
# Línea de referencia O(Δx)
ref_c1 = err_arr[-1] / dx_arr[-1]**1
ax_sp.loglog(dx_arr, ref_c1 * dx_arr**1, ':', color='#888780', lw=1.5,
             label='Pendiente O(Δx¹)')
ax_sp.set_xlabel('Δx [m]', fontsize=11)
ax_sp.set_ylabel('||e||_L²', fontsize=11)
ax_sp.set_title(f'Error L² vs Δx  (r={r_cfl}, t={t_samp:.1f}s)', fontsize=10)
ax_sp.legend(fontsize=10); ax_sp.grid(True, alpha=0.4, which='both')
# Anotar órdenes
for i in range(len(orders)):
    xm = np.sqrt(dx_arr[i] * dx_arr[i+1])
    ym = np.sqrt(err_arr[i] * err_arr[i+1])
    ax_sp.annotate(f'{orders[i]:.2f}', (xm, ym),
                   textcoords='offset points', xytext=(6, 3),
                   fontsize=8, color='#378ADD')

# ── Figura 1b: Convergencia temporal ──
ax_t = axes1[1]
ax_t.loglog(dt_arr, err_t_arr, 's-', color='#7F77DD', lw=2, ms=7,
            label='Error L² numérico')
ref_ct2 = err_t_arr[-1] / dt_arr[-1]**2
ax_t.loglog(dt_arr, ref_ct2 * dt_arr**2, '--', color='#D85A30', lw=1.8,
            label='Pendiente O(Δt²)')
ax_t.set_xlabel('Δt [s]', fontsize=11)
ax_t.set_ylabel('||e||_L²', fontsize=11)
ax_t.set_title(f'Error L² vs Δt  (N={N_fixed}, t={t_samp:.1f}s)', fontsize=10)
ax_t.legend(fontsize=10); ax_t.grid(True, alpha=0.4, which='both')

plt.tight_layout()
if args.guardar:
    plt.savefig('errores_convergencia.png', dpi=150, bbox_inches='tight')
    print("Guardado: errores_convergencia.png")
else:
    print("No se guardó: errores_convergencia.png")
plt.show()


# ── Figura 2: Error vs tiempo para distintos N ──
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
ax2.semilogy(t_rec40,  err_rec40,  color='#D85A30', lw=1.8, label='N = 40')
ax2.semilogy(t_rec80,  err_rec80,  color='#378ADD', lw=1.8, label='N = 80')
ax2.semilogy(t_rec120, err_rec120, color='#1D9E75', lw=1.8, label='N = 120')
ax2.set_xlabel('t [s]', fontsize=11)
ax2.set_ylabel('||e(t)||_L²  (escala log)', fontsize=11)
ax2.set_title('Evolución temporal del error L²  –  Distintos refinamientos (r = 0.5)\n'
              'CC: fijo (x=0) | libre (x=L)',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.4, which='both')
plt.tight_layout()
if args.guardar:
    plt.savefig('errores_vs_tiempo.png', dpi=150, bbox_inches='tight')
    print("Guardado: errores_vs_tiempo.png")
else:
    print("No se guardó: errores_vs_tiempo.png")
plt.show()


# ── Figura 3: Error de truncamiento – representación gráfica ──
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
fig3.suptitle('Error de truncamiento de la semi-discretización espacial\n'
              'D.F. centradas 2º orden:  d²u/dx² ≈ (u_{i-1}-2u_i+u_{i+1})/Δx² + O(Δx²)',
              fontsize=11, fontweight='bold')

t_fix = 0.3
x_fine = np.linspace(0, L, 500)
u_fine = u_analitica(x_fine, t_fix)
d2u_exact = -(np.pi/(2*L))**2 * u_fine    # d²/dx² de sin(πx/2L)·cos(...)

for ax, Nplot, col in zip(axes3, [20, 80], ['#D85A30', '#1D9E75']):
    dx_p = L / (Nplot - 1)
    xp   = np.linspace(0, L, Nplot)
    up   = u_analitica(xp, t_fix)
    d2u_num  = (np.roll(up, -1) - 2*up + np.roll(up, 1)) / dx_p**2
    d2u_num[0]  = (up[1] - 2*up[0]) / dx_p**2   # ajuste extremo fijo
    d2u_num[-1] = (up[-2] - up[-1]) / dx_p**2   # ajuste extremo libre
    d2u_exact_p = -(np.pi/(2*L))**2 * up

    ax.plot(x_fine, d2u_exact, 'k-', lw=2, label='Analítica  d²u/dx²')
    ax.plot(xp, d2u_num, 'o--', color=col, ms=4, lw=1.5, label=f'D.F. 2º ord.  N={Nplot}')
    ax.fill_between(xp, d2u_exact_p, d2u_num, alpha=0.25, color=col, label='Error puntual')
    ax.set_title(f'N = {Nplot}  (Δx = {dx_p:.4f} m)', fontsize=10)
    ax.set_xlabel('x [m]', fontsize=9); ax.set_ylabel('d²u/dx²', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.axvline(0, color='crimson', lw=1.5, ls='--', alpha=0.5)
    ax.axvline(L, color='forestgreen', lw=1.5, ls='--', alpha=0.5)

plt.tight_layout()
if args.guardar:
    plt.savefig('error_truncamiento.png', dpi=150, bbox_inches='tight')
    print("Guardado: error_truncamiento.png")
else:
    print("No se guardó: error_truncamiento.png")
plt.show()

print("\n=== Resumen de convergencia espacial ===")
print(f"{'N':>6}  {'Δx':>10}  {'Error L²':>12}  {'Orden':>8}")
for i, (N, dx_i, err) in enumerate(zip(N_list, dx_list, errors_2nd)):
    ord_str = f'{orders[i-1]:.3f}' if i > 0 else '–'
    print(f"{N:>6}  {dx_i:>10.6f}  {err:>12.4e}  {ord_str:>8}")
