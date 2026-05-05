"""
5_energia_casos_latigo.py  –  Conservación de energía y casos de prueba
=========================================================================
Puntos 6-7 del trabajo:
  • Conservación de la energía discreta (cinética + potencial)
  • Comparación de energía: Leapfrog vs Crank-Nicolson
  • Casos de prueba:
      A) Reflexión en el extremo libre (amplificación de 2×)
      B) Focalización del impulso ("chasquido del látigo")
      C) Propagación de múltiples rebotes
  • Perfil de velocidad de la punta (extremo libre) vs tiempo

Ejecutar:
    python 5_energia_casos_latigo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import (
    solver_explicito, solver_crank_nicolson,
    impulso_gaussiano, impulso_sinusoidal, impulso_triangular,
    energia_discreta, L, C, RHO
)

# ─────────────────────────────────────────────────────────────────
#  Parámetros comunes
# ─────────────────────────────────────────────────────────────────
N    = 120
dx   = L / (N - 1)
r    = 0.90
dt   = r * dx / C
T_MAX = 6.0

# ─────────────────────────────────────────────────────────────────
#  Función para extraer energía de todos los frames
# ─────────────────────────────────────────────────────────────────
def calcular_energia_frames(frames, dt_sim, dx_sim, c):
    t_arr, Ek_arr, Ep_arr = [], [], []
    for i in range(1, len(frames)):
        t_i,  u_i  = frames[i]
        _,    u_im1 = frames[i - 1]
        Ek, Ep = energia_discreta(u_i, u_im1, dt_sim, dx_sim, c)
        t_arr.append(t_i)
        Ek_arr.append(Ek)
        Ep_arr.append(Ep)
    return (np.array(t_arr),
            np.array(Ek_arr),
            np.array(Ep_arr),
            np.array(Ek_arr) + np.array(Ep_arr))


# ─────────────────────────────────────────────────────────────────
#  Ejecutar simulaciones
# ─────────────────────────────────────────────────────────────────
print("Ejecutando simulaciones...")
fr_lf, x, _ = solver_explicito(N=N, c=C, dt=dt, t_max=T_MAX,
                                func_impulso=impulso_gaussiano)
fr_cn, _, _  = solver_crank_nicolson(N=N, c=C, dt=dt, t_max=T_MAX,
                                      func_impulso=impulso_gaussiano)

t_lf = np.array([f[0] for f in fr_lf])
t_cn = np.array([f[0] for f in fr_cn])
U_lf = np.array([f[1] for f in fr_lf])
U_cn = np.array([f[1] for f in fr_cn])

print(f"  Frames Leapfrog: {len(fr_lf)},  Frames CN: {len(fr_cn)}")


# ─────────────────────────────────────────────────────────────────
#  Figura 1: Conservación de energía Leapfrog vs CN
# ─────────────────────────────────────────────────────────────────
t_e, Ek_lf, Ep_lf, Et_lf = calcular_energia_frames(fr_lf, dt, dx, C)
t_ec, Ek_cn, Ep_cn, Et_cn = calcular_energia_frames(fr_cn, dt, dx, C)

E0_lf = Et_lf[0] if len(Et_lf) > 0 else 1.0
E0_cn = Et_cn[0] if len(Et_cn) > 0 else 1.0

fig1, axes1 = plt.subplots(2, 2, figsize=(13, 8))
fig1.suptitle('Conservación de energía  –  Ecuación de ondas (Látigo)\n'
              f'N={N},  r={r:.2f},  CC: fijo | libre',
              fontsize=11, fontweight='bold')

# ── Leapfrog ──
ax = axes1[0, 0]
ax.plot(t_e, Ek_lf / E0_lf, color='#D85A30', lw=1.8, label='Energía cinética Eₖ')
ax.plot(t_e, Ep_lf / E0_lf, color='#378ADD', lw=1.8, label='Energía potencial Eₚ')
ax.plot(t_e, Et_lf / E0_lf, color='#444441', lw=2.5, label='Energía total E_tot')
ax.axhline(1.0, color='#1D9E75', lw=1.5, ls='--', label='E₀ (referencia)')
ax.set_xlabel('t [s]', fontsize=10); ax.set_ylabel('E / E₀', fontsize=10)
ax.set_title('Leapfrog (explícito)', fontsize=10, fontweight='bold', color='#378ADD')
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.35); ax.set_xlim(0, T_MAX)

# ── CN ──
ax = axes1[0, 1]
ax.plot(t_ec, Ek_cn / E0_cn, color='#D85A30', lw=1.8, label='Energía cinética Eₖ')
ax.plot(t_ec, Ep_cn / E0_cn, color='#378ADD', lw=1.8, label='Energía potencial Eₚ')
ax.plot(t_ec, Et_cn / E0_cn, color='#444441', lw=2.5, label='Energía total E_tot')
ax.axhline(1.0, color='#1D9E75', lw=1.5, ls='--', label='E₀ (referencia)')
ax.set_xlabel('t [s]', fontsize=10); ax.set_ylabel('E / E₀', fontsize=10)
ax.set_title('Crank-Nicolson (implícito)', fontsize=10, fontweight='bold', color='#7F77DD')
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.35); ax.set_xlim(0, T_MAX)

# ── Error de energía ──
ax = axes1[1, 0]
ax.semilogy(t_e,  np.abs(Et_lf / E0_lf - 1), color='#378ADD', lw=1.8,
            label='Leapfrog |ΔE/E₀|')
ax.semilogy(t_ec, np.abs(Et_cn / E0_cn - 1), color='#7F77DD', lw=1.8, ls='--',
            label='Crank-Nicolson |ΔE/E₀|')
ax.set_xlabel('t [s]', fontsize=10)
ax.set_ylabel('Error relativo de energía (log)', fontsize=10)
ax.set_title('Deriva de energía total  |E(t)/E₀ – 1|', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.35, which='both'); ax.set_xlim(0, T_MAX)

# ── Energía total durante el impulso y tras apagarlo ──
ax = axes1[1, 1]
# Impulso gaussiano activo principalmente en t ∈ [0.1, 0.45]
t_impulso = np.linspace(0, 0.6, 200)
imp_vals  = np.array([impulso_gaussiano(ti) for ti in t_impulso])
ax2 = ax.twinx()
ax.plot(t_e, Et_lf / E0_lf,  color='#378ADD', lw=2, label='Energía total (Leapfrog)')
ax2.plot(t_impulso, imp_vals, color='crimson', lw=1.5, ls='--', alpha=0.7,
         label='Impulso en x=0')
ax.set_xlabel('t [s]', fontsize=10)
ax.set_ylabel('E_tot / E₀', fontsize=10)
ax2.set_ylabel('u(0,t)  [impulso]', fontsize=10, color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')
ax.set_title('Inyección de energía por el impulso de CC', fontsize=10)
ax.grid(True, alpha=0.35)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc='upper right')
ax.set_xlim(0, T_MAX)

plt.tight_layout()
plt.savefig('energia_latigo.png', dpi=150, bbox_inches='tight')
print("Guardado: energia_latigo.png")
plt.show()


# ─────────────────────────────────────────────────────────────────
#  Figura 2: Velocidad de la punta (extremo libre) vs tiempo
#  → El "chasquido" corresponde al máximo de |∂u/∂t| en x=L
# ─────────────────────────────────────────────────────────────────
vel_punta = []
for i in range(1, len(fr_lf)):
    _, u_i   = fr_lf[i]
    _, u_im1 = fr_lf[i - 1]
    vel_punta.append((u_i[-1] - u_im1[-1]) / dt)
vel_punta = np.array(vel_punta)
t_vel = t_lf[1:]

u_punta = U_lf[:, -1]  # desplazamiento de la punta

fig2 = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(3, 2, figure=fig2)
fig2.suptitle('Dinámica de la punta del látigo  –  "Chasquido"\n'
              'Extremo libre (x=L, Neumann) con impulso en x=0',
              fontsize=11, fontweight='bold')

# ── Desplazamiento punta ──
ax_u = fig2.add_subplot(gs[0, :])
ax_u.plot(t_lf, u_punta, color='#1D9E75', lw=2, label='u(L, t) – desplazamiento punta')
ax_u.fill_between(t_lf, u_punta, alpha=0.15, color='#1D9E75')
ax_u.axhline(0, color='gray', lw=0.5); ax_u.grid(True, alpha=0.35)
ax_u.set_xlabel('t [s]', fontsize=10); ax_u.set_ylabel('u(L, t)', fontsize=10)
ax_u.set_title('Desplazamiento del extremo libre (punta del látigo)', fontsize=10)
ax_u.legend(fontsize=9); ax_u.set_xlim(0, T_MAX)

# ── Velocidad de la punta ──
ax_v = fig2.add_subplot(gs[1, :])
ax_v.plot(t_vel, vel_punta, color='#D85A30', lw=1.8, label='∂u/∂t (L, t) – velocidad punta')
# Marcar máximo
idx_max = np.argmax(np.abs(vel_punta))
ax_v.scatter(t_vel[idx_max], vel_punta[idx_max], color='crimson', s=80, zorder=5,
             label=f'Vel. max = {vel_punta[idx_max]:.3f} m/s  (t={t_vel[idx_max]:.3f}s)')
ax_v.axhline(0, color='gray', lw=0.5); ax_v.grid(True, alpha=0.35)
ax_v.set_xlabel('t [s]', fontsize=10); ax_v.set_ylabel('∂u/∂t (L, t)', fontsize=10)
ax_v.set_title('Velocidad del extremo libre  –  "Chasquido" = máximo de velocidad', fontsize=10)
ax_v.legend(fontsize=9); ax_v.set_xlim(0, T_MAX)

# ── Perfil espacial en el instante del chasquido ──
idx_crack = np.argmin(np.abs(t_lf - t_vel[idx_max]))
ax_cr = fig2.add_subplot(gs[2, 0])
ax_cr.plot(x, U_lf[idx_crack], color='#D85A30', lw=2.5)
ax_cr.fill_between(x, U_lf[idx_crack], alpha=0.2, color='#D85A30')
ax_cr.axvline(x[-1], color='forestgreen', lw=2, ls='--', label='Punta libre')
ax_cr.scatter([x[-1]], [U_lf[idx_crack, -1]], color='forestgreen', s=80, zorder=5)
ax_cr.set_xlabel('x [m]', fontsize=10); ax_cr.set_ylabel('u(x, t*)', fontsize=10)
ax_cr.set_title(f'Perfil en el chasquido  (t* = {t_lf[idx_crack]:.3f} s)', fontsize=10)
ax_cr.legend(fontsize=9); ax_cr.grid(True, alpha=0.35)
ax_cr.set_xlim(0, L)

# ── Espectro de frecuencias de u(L, t) ──
ax_fft = fig2.add_subplot(gs[2, 1])
fft_u = np.fft.rfft(u_punta)
freqs = np.fft.rfftfreq(len(u_punta), d=t_lf[1] - t_lf[0])
potencia = np.abs(fft_u)**2
ax_fft.plot(freqs[:len(freqs)//4], potencia[:len(freqs)//4],
            color='#7F77DD', lw=1.5)
ax_fft.set_xlabel('Frecuencia [Hz]', fontsize=10)
ax_fft.set_ylabel('|FFT(u(L,t))|²', fontsize=10)
ax_fft.set_title('Espectro de frecuencias  –  Punta del látigo', fontsize=10)
ax_fft.grid(True, alpha=0.35)
# Frecuencias propias (2n-1)c/(4L)
for n in range(1, 6):
    fn = (2*n - 1) * C / (4 * L)
    ax_fft.axvline(fn, color='crimson', lw=0.8, ls='--', alpha=0.6,
                   label=f'f_{n}={fn:.2f}Hz' if n <= 3 else '')
ax_fft.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig('chasquido_punta.png', dpi=150, bbox_inches='tight')
print("Guardado: chasquido_punta.png")
plt.show()


# ─────────────────────────────────────────────────────────────────
#  Figura 3: Casos de prueba – Distintos impulsos
# ─────────────────────────────────────────────────────────────────
casos = [
    ('Gaussiano',   impulso_gaussiano,   '#378ADD'),
    ('Senoidal',    impulso_sinusoidal,  '#1D9E75'),
    ('Triangular',  impulso_triangular,  '#D85A30'),
]

fig3, axes3 = plt.subplots(3, 3, figsize=(14, 10))
fig3.suptitle('Casos de prueba  –  Distintas formas de impulso en el extremo fijo\n'
              'Diagrama espacio-tiempo  (fila = caso,  columna = t uniforme)',
              fontsize=11, fontweight='bold')

for row, (nombre, func, col) in enumerate(casos):
    fr_i, x_i, _ = solver_explicito(N=N, c=C, dt=dt, t_max=T_MAX,
                                     func_impulso=func)
    t_i  = np.array([f[0] for f in fr_i])
    U_i  = np.array([f[1] for f in fr_i])
    t_snaps = np.linspace(0.2, T_MAX * 0.8, 3)

    for col_idx, ts in enumerate(t_snaps):
        ax = axes3[row, col_idx]
        idx = np.argmin(np.abs(t_i - ts))
        ax.fill_between(x_i, U_i[idx], alpha=0.2, color=col)
        ax.plot(x_i, U_i[idx], color=col, lw=2)
        ax.axvline(0, color='crimson',     lw=1.5, ls='--', alpha=0.5)
        ax.axvline(L, color='forestgreen', lw=1.5, ls='--', alpha=0.5)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlim(0, L); ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x [m]', fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(f'{nombre}\nu(x,t)', fontsize=8, color=col)
        ax.set_title(f't = {t_i[idx]:.2f} s', fontsize=9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('casos_prueba_latigo.png', dpi=150, bbox_inches='tight')
print("Guardado: casos_prueba_latigo.png")
plt.show()


# ─────────────────────────────────────────────────────────────────
#  Resumen numérico
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RESUMEN NUMÉRICO – Conservación de energía")
print("="*55)
print(f"  Esquema Leapfrog: Δ(E/E₀) max = {np.max(np.abs(Et_lf/E0_lf - 1)):.2e}")
print(f"  Esquema CN:       Δ(E/E₀) max = {np.max(np.abs(Et_cn/E0_cn - 1)):.2e}")
print(f"\nChasquido del látigo:")
print(f"  Velocidad máxima punta = {np.max(np.abs(vel_punta)):.4f} m/s")
print(f"  en t* = {t_vel[np.argmax(np.abs(vel_punta))]:.4f} s")
print(f"  c física = {C:.2f} m/s  →  factor = {np.max(np.abs(vel_punta))/C:.2f}×c")
fn1 = C / (4 * L)
print(f"\nFrecuencia fundamental (CC fijo-libre): f₁ = c/(4L) = {fn1:.4f} Hz")
print(f"Armónicos: f_n = (2n-1)·f₁  →  {[round((2*n-1)*fn1,3) for n in range(1,5)]} Hz")
