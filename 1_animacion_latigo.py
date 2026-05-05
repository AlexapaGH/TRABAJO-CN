"""
1_animacion_latigo.py  –  Animación 2D de la propagación de onda en el látigo
===============================================================================
Muestra la evolución temporal u(x, t) con:
  • Extremo FIJO (x=0, mango): impulso gaussiano prescrito
  • Extremo LIBRE (x=L, punta): condición de Neumann ∂u/∂x = 0

Ejecutar:
    python 1_animacion_latigo.py

Genera además:
  - snapshot de fotogramas representativos
  - figura 2D del látigo animado (guardar como GIF o MP4 descomentando save)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from utils import (
    solver_explicito, impulso_gaussiano,
    impulso_sinusoidal, impulso_triangular, L, C
)

# ───────────────────────────────────────────────────────────────
#  Parámetros de simulación
# ───────────────────────────────────────────────────────────────
N      = 120          # Número de nodos espaciales
r      = 0.90         # Número de CFL  (r = c·Δt/Δx ≤ 1 para estabilidad)
dx     = L / (N - 1)
dt     = r * dx / C
T_MAX  = 4.0          # Tiempo total [s]

print(f"Parámetros:\n  N={N}, dx={dx:.5f} m, dt={dt:.6f} s, r={r:.2f}")
print(f"  Pasos totales: {int(T_MAX/dt)}, CFL={r:.2f}")

# ───────────────────────────────────────────────────────────────
#  Resolver con impulso gaussiano
# ───────────────────────────────────────────────────────────────
frames, x, r_eff = solver_explicito(
    N=N, c=C, dt=dt, t_max=T_MAX,
    func_impulso=impulso_gaussiano
)

t_arr = np.array([f[0] for f in frames])
U_mat = np.array([f[1] for f in frames])   # shape (n_frames, N)

# ───────────────────────────────────────────────────────────────
#  Figura 1: Snapshots en tiempos representativos
# ───────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
fig1.suptitle('Ecuación de ondas 1D  –  Látigo\n'
              'CC: extremo fijo (impulso gaussiano) | extremo libre (Neumann)',
              fontsize=12, fontweight='bold')

t_snaps = np.linspace(0, T_MAX * 0.9, 6)
for ax, ts in zip(axes.flat, t_snaps):
    idx = np.argmin(np.abs(t_arr - ts))
    u   = U_mat[idx]
    t   = t_arr[idx]

    ax.fill_between(x, u, alpha=0.25, color='royalblue')
    ax.plot(x, u, color='royalblue', lw=2)

    # Marcadores de CC
    ax.axvline(0, color='crimson', lw=1.5, ls='--', alpha=0.7)
    ax.axvline(L, color='forestgreen', lw=1.5, ls='--', alpha=0.7)
    ax.scatter([0], [u[0]], color='crimson', zorder=5, s=40)

    # Indicador de "mango" y "punta"
    ax.text(0.03, 0.92, 'Mango\n(fijo)',
            transform=ax.transAxes, fontsize=7, color='crimson',
            va='top', ha='left')
    ax.text(0.97, 0.92, 'Punta\n(libre)',
            transform=ax.transAxes, fontsize=7, color='forestgreen',
            va='top', ha='right')

    ax.set_title(f't = {t:.3f} s', fontsize=10)
    ax.set_xlim(0, L)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('u(x,t)', fontsize=9)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('snapshots_latigo.png', dpi=150, bbox_inches='tight')
print("Guardado: snapshots_latigo.png")
plt.show()

# ───────────────────────────────────────────────────────────────
#  Figura 2: Diagrama espacio-tiempo (x-t)
# ───────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
extent = [x[0], x[-1], t_arr[0], t_arr[-1]]
im = ax2.imshow(
    U_mat, aspect='auto', origin='lower', extent=extent,
    cmap='RdBu_r', vmin=-1.1, vmax=1.1, interpolation='bilinear'
)
plt.colorbar(im, ax=ax2, label='u(x, t)  [desplazamiento adimensional]')
ax2.set_xlabel('x [m]', fontsize=11)
ax2.set_ylabel('t [s]',  fontsize=11)
ax2.set_title('Diagrama espacio-tiempo  –  Látigo 1D\n'
              'Propagación, reflexión y efecto de extremo libre',
              fontsize=11, fontweight='bold')
ax2.axvline(0, color='crimson',     lw=1.5, ls='--', label='Extremo fijo (mango)')
ax2.axvline(L, color='forestgreen', lw=1.5, ls='--', label='Extremo libre (punta)')
ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('diagrama_xt_latigo.png', dpi=150, bbox_inches='tight')
print("Guardado: diagrama_xt_latigo.png")
plt.show()

# ───────────────────────────────────────────────────────────────
#  Figura 3: Animación 2D del látigo
# ───────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.set_xlim(0, L)
ax3.set_ylim(-1.3, 1.3)
ax3.set_xlabel('x [m]', fontsize=11)
ax3.set_ylabel('Desplazamiento u(x, t)', fontsize=11)
ax3.set_title('Animación – Látigo 1D  (CC: fijo | libre)', fontsize=11, fontweight='bold')
ax3.axhline(0, color='gray', lw=0.5, ls=':')
ax3.grid(True, alpha=0.3)

# Indicadores de extremos
ax3.axvline(0, color='crimson',     lw=2, ls='--', alpha=0.6)
ax3.axvline(L, color='forestgreen', lw=2, ls='--', alpha=0.6)
ax3.text(0.01, 1.15, 'Mango (fijo)',
         color='crimson', fontsize=9, fontstyle='italic')
ax3.text(0.88, 1.15, 'Punta (libre)',
         color='forestgreen', fontsize=9, fontstyle='italic')

# Representación 2D tipo látigo: el eje x es la longitud del látigo
# y la amplitud representa el desplazamiento transversal
fill   = ax3.fill_between(x, np.zeros(N), alpha=0.25, color='royalblue')
line,  = ax3.plot([], [], color='royalblue', lw=2.5)
punto_mango, = ax3.plot([], [], 'o', color='crimson', ms=8, zorder=5)
punto_punta, = ax3.plot([], [], 'o', color='forestgreen', ms=8, zorder=5)
time_text = ax3.text(0.78, 0.90, '', transform=ax3.transAxes,
                     fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.7))


def init_anim():
    line.set_data([], [])
    punto_mango.set_data([], [])
    punto_punta.set_data([], [])
    time_text.set_text('')
    return line, punto_mango, punto_punta, time_text


def update_anim(i):
    global fill
    t_i = t_arr[i]
    u_i = U_mat[i]
    line.set_data(x, u_i)
    punto_mango.set_data([x[0]],  [u_i[0]])
    punto_punta.set_data([x[-1]], [u_i[-1]])
    time_text.set_text(f't = {t_i:.3f} s\nr = {r_eff:.2f}')
    # Eliminar el relleno anterior y crear uno nuevo
    fill.remove()
    fill = ax3.fill_between(x, u_i, alpha=0.20, color='royalblue')
    return line, punto_mango, punto_punta, time_text, fill


skip   = max(1, len(frames) // 200)   # ~200 fotogramas en la animación
frames_anim = list(range(0, len(frames), skip))

ani = animation.FuncAnimation(
    fig3, update_anim, frames=frames_anim,
    init_func=init_anim, interval=25, blit=False, repeat=True
)

# Descomenta para guardar (requiere ffmpeg o pillow):
# ani.save('animacion_latigo.mp4', writer='ffmpeg', fps=30, dpi=120)
# ani.save('animacion_latigo.gif', writer='pillow', fps=25)
print("Animación lista. Cierra la ventana para continuar.")
plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────
#  Figura 4: Comparación de formas de impulso inicial
# ───────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
fig4.suptitle('Efecto de la forma del impulso en la propagación  (t = 0.5 s)',
              fontsize=11, fontweight='bold')

impulsos = [
    ('Gaussiano',   impulso_gaussiano),
    ('Senoidal',    impulso_sinusoidal),
    ('Triangular',  impulso_triangular),
]

t_target = 0.5
for ax, (nombre, func) in zip(axes4, impulsos):
    fr, x_, _ = solver_explicito(N=N, c=C, dt=dt, t_max=t_target,
                                  func_impulso=func)
    t_vals = np.array([f[0] for f in fr])
    U_vals = np.array([f[1] for f in fr])

    # Dibuja evolución con degradado de color
    for k in range(0, len(fr), max(1, len(fr)//10)):
        alpha = 0.15 + 0.75 * k / len(fr)
        color = plt.cm.Blues(0.4 + 0.5 * k / len(fr))
        ax.plot(x_, U_vals[k], color=color, lw=1.2, alpha=alpha)

    ax.plot(x_, U_vals[-1], color='royalblue', lw=2.2, label=f't={t_target:.1f}s')
    ax.axvline(0, color='crimson',     lw=1.5, ls='--', alpha=0.6)
    ax.axvline(L, color='forestgreen', lw=1.5, ls='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_title(nombre, fontsize=11)
    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('u(x, t)', fontsize=9)
    ax.set_xlim(0, L); ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.3)

    # Señal de impulso en el tiempo
    t_vec = np.linspace(0, t_target, 300)
    imp_vals = np.array([func(ti) for ti in t_vec])
    ax_inset = ax.inset_axes([0.55, 0.55, 0.42, 0.38])
    ax_inset.plot(t_vec, imp_vals, color='crimson', lw=1.5)
    ax_inset.set_title('Impulso(t)', fontsize=7, pad=2)
    ax_inset.set_xlabel('t', fontsize=7); ax_inset.tick_params(labelsize=6)
    ax_inset.grid(True, alpha=0.3); ax_inset.axhline(0, color='gray', lw=0.5)

plt.tight_layout()
plt.savefig('comparacion_impulsos.png', dpi=150, bbox_inches='tight')
print("Guardado: comparacion_impulsos.png")
plt.show()
