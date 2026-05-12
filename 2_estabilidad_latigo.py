"""
2_estabilidad_latigo.py  –  Análisis de estabilidad asintótica
==============================================================
Contenido (punto 5 del trabajo):
  A) Criterio de Von Neumann para el esquema explícito
  B) Demostración visual: soluciones estables vs inestables para distintos Δt
  C) Mapa CFL: amplificación máxima |G(r, kΔx)| en el plano (r, kΔx)
  D) Comparación explícito vs Crank-Nicolson (incond. estable) a r > 1

Ejecutar:
    python 2_estabilidad_latigo.py [--guardar] o [--no-guardar]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import solver_explicito, solver_crank_nicolson, impulso_gaussiano, L, C

# ───────────────────────────────────────────────────────────────
#  Argumentos de línea de comandos
# ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Análisis de estabilidad del látigo 1D',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('--guardar', action='store_true', default=False,
                    help='Guardar todos los archivos (valor por defecto)')
args = parser.parse_args()

# ───────────────────────────────────────────────────────────────
#  Parámetros
# ───────────────────────────────────────────────────────────────
N      = 80
dx     = L / (N - 1)
dt_crit = dx / C        # Δt crítico (r = 1)

T_DEMO = 2.0             # Tiempo de demostración
r_vals = [0.5, 0.9, 1.0, 1.1, 1.5]   # Factores de r a comparar


# ───────────────────────────────────────────────────────────────
#  A) Función de amplificación de Von Neumann (esquema explícito)
#     |G(r, ξ)|  donde ξ = kΔx ∈ [0, π]
# ───────────────────────────────────────────────────────────────
def amplificacion_max(r):
    """Amplificación máxima sobre todos los modos de Fourier.
    Ec. característica: G² - 2αG + 1 = 0, α = 1 - 2r²sin²(kΔx/2).
    |G| = 1 si |α| ≤ 1 (estable);  |G| = |α| + √(α²-1) si |α| > 1.
    """
    xi    = np.linspace(0, np.pi, 500)
    mu    = r * np.sin(xi / 2)          # μ = r·sin(kΔx/2)
    alpha = 1.0 - 2.0 * mu**2
    G     = np.where(
        np.abs(alpha) <= 1.0,
        1.0,
        np.abs(alpha) + np.sqrt(np.maximum(alpha**2 - 1.0, 0.0))
    )
    return np.max(G)


# ───────────────────────────────────────────────────────────────
#  B) Soluciones estables e inestables
# ───────────────────────────────────────────────────────────────
print("Ejecutando simulaciones de estabilidad...")
resultados = {}
for r_fac in r_vals:
    dt_i = r_fac * dt_crit
    try:
        fr, x, r_eff = solver_explicito(N=N, c=C, dt=dt_i,
                                         t_max=T_DEMO,
                                         func_impulso=impulso_gaussiano)
        # Detectar divergencia
        U = np.array([f[1] for f in fr])
        max_amp = np.nanmax(np.abs(U))
        diverge = max_amp > 50 or np.any(np.isnan(U)) or np.any(np.isinf(U))
        resultados[r_fac] = {'frames': fr, 'x': x, 'U': U,
                             'max_amp': max_amp, 'diverge': diverge}
    except Exception as e:
        resultados[r_fac] = {'diverge': True, 'max_amp': np.inf}
    print(f"  r={r_fac:.2f}  →  max|u| = {resultados[r_fac].get('max_amp', np.inf):.2e}"
          f"  {'⚠ INESTABLE' if resultados[r_fac]['diverge'] else '✓ estable'}")


# ───────────────────────────────────────────────────────────────
#  Figura 1: Soluciones estables vs inestables al tiempo t=T_DEMO
# ───────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, len(r_vals), figsize=(16, 4), sharey=False)
fig1.suptitle(
    'Esquema explícito Leapfrog  –  Soluciones para distintos r = cΔt/Δx\n'
    f'N = {N},  t = {T_DEMO:.1f} s',
    fontsize=11, fontweight='bold'
)

colors_r = ['#1a6b3c', '#1D9E75', '#888780', '#D85A30', '#99200f']
for ax, r_fac, color in zip(axes, r_vals, colors_r):
    res = resultados[r_fac]
    U   = res.get('U', None)
    x_  = res.get('x', np.linspace(0, L, N))
    est = 'INESTABLE ⚠' if res['diverge'] else 'Estable ✓'
    col = '#D85A30' if res['diverge'] else '#1D9E75'

    if U is not None and not res['diverge']:
        u_last = U[-1]
        ax.fill_between(x_, u_last, alpha=0.2, color=color)
        ax.plot(x_, u_last, color=color, lw=2)
        ax.set_ylim(-1.5, 1.5)
    elif U is not None:
        u_clip = np.clip(U[-1], -5, 5)
        ax.plot(x_, u_clip, color=color, lw=1.5, ls='--')
        ax.set_ylim(-5, 5)

    ax.axvline(0, color='crimson',     lw=1.2, ls='--', alpha=0.6)
    ax.axvline(L, color='forestgreen', lw=1.2, ls='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_title(f'r = {r_fac:.2f}\n{est}', fontsize=10, color=col, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('u(x, t)', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
if args.guardar:
    plt.savefig('estabilidad_soluciones.png', dpi=150, bbox_inches='tight')
    print("Guardado: estabilidad_soluciones.png")
else:
    print("No se guardó: estabilidad_soluciones.png")
plt.show()


# ───────────────────────────────────────────────────────────────
#  Figura 2: Mapa de amplificación |G(r, kΔx)| – Región de estabilidad
# ───────────────────────────────────────────────────────────────
r_range  = np.linspace(0.0, 2.0, 400)
xi_range = np.linspace(0.0, np.pi, 400)
R, Xi    = np.meshgrid(r_range, xi_range)

mu    = R * np.sin(Xi / 2)             # μ = r·sin(kΔx/2)
alpha = 1.0 - 2.0 * mu**2
G     = np.where(
    np.abs(alpha) <= 1.0,
    1.0,
    np.abs(alpha) + np.sqrt(np.maximum(alpha**2 - 1.0, 0.0))
)

fig2, ax2 = plt.subplots(figsize=(9, 5))
levels = np.linspace(0, 4, 40)
cf = ax2.contourf(r_range, xi_range / np.pi, G,
                  levels=levels, cmap='RdYlGn_r', extend='max')
plt.colorbar(cf, ax=ax2, label='|G(r, kΔx)|  [factor de amplificación por paso]')
ax2.contour(r_range, xi_range / np.pi, G,
            levels=[1.0], colors='white', linewidths=2.5)
ax2.axvline(1.0, color='white', lw=2, ls='--', label='r = 1  (criterio CFL)')

# Anotaciones
ax2.text(0.50, 0.5, 'ZONA ESTABLE\n|G| ≤ 1',
         color='white', fontsize=12, ha='center', va='center', fontweight='bold')
ax2.text(1.50, 0.5, 'ZONA INESTABLE\n|G| > 1',
         color='white', fontsize=12, ha='center', va='center', fontweight='bold')

ax2.set_xlabel('r = cΔt/Δx  (Número de Courant)', fontsize=11)
ax2.set_ylabel('kΔx / π  (Número de onda normalizado)', fontsize=11)
ax2.set_title('Mapa de estabilidad – Esquema explícito (Von Neumann)\n'
              'Línea blanca: frontera de estabilidad |G| = 1', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 1)
plt.tight_layout()
if args.guardar:
    plt.savefig('mapa_estabilidad.png', dpi=150, bbox_inches='tight')
    print("Guardado: mapa_estabilidad.png")
else:
    print("No se guardó: mapa_estabilidad.png")
plt.show()


# ───────────────────────────────────────────────────────────────
#  Figura 3: Amplificación máxima vs r  (explícito vs CN)
# ───────────────────────────────────────────────────────────────
r_vec   = np.linspace(0.0, 2.5, 300)
G_max_e = np.array([amplificacion_max(r) for r in r_vec])
G_max_cn = np.ones_like(r_vec)          # CN siempre |G| ≤ 1

fig3, ax3 = plt.subplots(figsize=(8, 4.5))
ax3.plot(r_vec, G_max_e,  color='#D85A30', lw=2.2, label='Explícito (Leapfrog / FTCS)')
ax3.plot(r_vec, G_max_cn, color='#378ADD', lw=2.2, ls='--', label='Crank-Nicolson  (|G| = 1 ∀r)')
ax3.axhline(1.0, color='gray', lw=1, ls=':')
ax3.axvline(1.0, color='gray', lw=1, ls=':', label='r = 1 (CFL crítico)')
ax3.fill_between(r_vec[r_vec <= 1], G_max_e[r_vec <= 1],
                 alpha=0.12, color='#1D9E75', label='Región estable')
ax3.fill_between(r_vec[r_vec > 1], G_max_e[r_vec > 1],
                 alpha=0.12, color='#D85A30')

ax3.set_xlabel('r = cΔt/Δx', fontsize=11)
ax3.set_ylabel('max|G(r)|  sobre todos los modos kΔx', fontsize=11)
ax3.set_title('Factor de amplificación máximo vs número de Courant r\n'
              'Esquema explícito (condicionalmente estable) vs Crank-Nicolson',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_xlim(0, 2.5); ax3.set_ylim(0, 4.5)
ax3.grid(True, alpha=0.35)
plt.tight_layout()
if args.guardar:
    plt.savefig('amplificacion_vs_r.png', dpi=150, bbox_inches='tight')
    print("Guardado: amplificacion_vs_r.png")
else:
    print("No se guardó: amplificacion_vs_r.png")
plt.show()


# ───────────────────────────────────────────────────────────────
#  Figura 4: Evolución temporal con r=1.2 → CN vs Explícito
# ───────────────────────────────────────────────────────────────
r_inest = 1.2
dt_i    = r_inest * dt_crit

fr_exp, x_e, _ = solver_explicito(N=N, c=C, dt=dt_i, t_max=T_DEMO,
                                    func_impulso=impulso_gaussiano)
fr_cn,  x_c, _ = solver_crank_nicolson(N=N, c=C, dt=dt_i, t_max=T_DEMO,
                                        func_impulso=impulso_gaussiano)

U_exp = np.clip(np.array([f[1] for f in fr_exp]), -5, 5)
U_cn  = np.array([f[1] for f in fr_cn])
t_exp = np.array([f[0] for f in fr_exp])
t_cn  = np.array([f[0] for f in fr_cn])

t_plot = [0.5, 1.0, 1.5, 2.0]
fig4, axes4 = plt.subplots(2, 4, figsize=(16, 6), sharey=False)
fig4.suptitle(f'r = {r_inest:.1f} (inestable para explícito)  –  Explícito vs Crank-Nicolson',
              fontsize=11, fontweight='bold')

for col, tp in enumerate(t_plot):
    idx_e = np.argmin(np.abs(t_exp - tp))
    idx_c = np.argmin(np.abs(t_cn  - tp))

    ax_e = axes4[0, col]
    ax_c = axes4[1, col]

    ax_e.plot(x_e, U_exp[idx_e], color='#D85A30', lw=2)
    ax_e.set_title(f'Explícito  t={tp:.1f}s', fontsize=9, color='#D85A30')
    ax_e.set_ylim(-5, 5); ax_e.set_xlabel('x [m]', fontsize=8)
    ax_e.axhline(0, color='gray', lw=0.5); ax_e.grid(True, alpha=0.3)

    ax_c.plot(x_c, U_cn[idx_c], color='#378ADD', lw=2)
    ax_c.set_title(f'Crank-Nicolson  t={tp:.1f}s', fontsize=9, color='#378ADD')
    ax_c.set_ylim(-1.5, 1.5); ax_c.set_xlabel('x [m]', fontsize=8)
    ax_c.axhline(0, color='gray', lw=0.5); ax_c.grid(True, alpha=0.3)

axes4[0, 0].set_ylabel('u(x,t)  [Explícito]', fontsize=9)
axes4[1, 0].set_ylabel('u(x,t)  [CN]', fontsize=9)
plt.tight_layout()
if args.guardar:
    plt.savefig('estabilidad_exp_vs_cn.png', dpi=150, bbox_inches='tight')
    print("Guardado: estabilidad_exp_vs_cn.png")
else:
    print("No se guardó: estabilidad_exp_vs_cn.png")
plt.show()

print("\nResumen de estabilidad:")
print(f"  Δx  = {dx:.5f} m")
print(f"  Δt_crit = {dt_crit:.6f} s  (r = 1)")
for r_f in r_vals:
    print(f"  r = {r_f:.2f}  →  {'INESTABLE' if resultados[r_f]['diverge'] else 'ESTABLE'}")
