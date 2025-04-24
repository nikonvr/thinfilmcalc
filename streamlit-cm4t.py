# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import io
from scipy.interpolate import make_interp_spline
import numba

@numba.jit(nopython=True, cache=True)
def calcul_M(pol, l_calc, theta_calc, Ni, ep_couche):
    alpha = np.sin(theta_calc)
    sqrt_term_sq = Ni**2 - alpha**2
    if np.real(sqrt_term_sq) < 0:
         sqrt_val = 1j * np.sqrt(-sqrt_term_sq)
    elif abs(np.real(sqrt_term_sq)) < 1e-12 :
          sqrt_val = 1e-6
    else:
         sqrt_val = np.sqrt(sqrt_term_sq)

    if pol == 'p':
         if abs(sqrt_val) < 1e-9:
             eta = np.inf + 0j
         else:
              eta = Ni**2 / sqrt_val
    else:
         eta = sqrt_val

    phi = (2 * np.pi / l_calc) * sqrt_val * ep_couche

    sin_phi_over_eta = 0j
    if abs(eta) > 1e-12:
        sin_phi_over_eta = (1j / eta) * np.sin(phi)
    elif abs(np.sin(phi)) > 1e-9:
         sin_phi_over_eta = np.inf + 0j

    M_layer = np.array([[np.cos(phi), sin_phi_over_eta],
                      [1j * eta * np.sin(phi), np.cos(phi)]], dtype=np.complex128)
    return M_layer, eta

@numba.jit(nopython=True, cache=True)
def calcul_etats(pol, theta_calc, n_inc, n_sub):
     alpha = np.sin(theta_calc)
     etainc, etasub = 0j, 0j

     sqrt_term_inc_sq = n_inc**2 - alpha**2
     if np.real(sqrt_term_inc_sq) < 0 : sqrt_val_inc = 1j * np.sqrt(-sqrt_term_inc_sq)
     elif abs(np.real(sqrt_term_inc_sq)) < 1e-12: sqrt_val_inc = 1e-6
     else: sqrt_val_inc = np.sqrt(sqrt_term_inc_sq)

     n_sub_complex = np.complex128(n_sub)

     if pol == 'p':
          if abs(sqrt_val_inc) < 1e-9:
              etainc = np.inf + 0j
          else:
              etainc = n_inc**2 / sqrt_val_inc
     else:
          etainc = sqrt_val_inc

     sqrt_term_sub_sq = n_sub_complex**2 - alpha**2
     if np.real(sqrt_term_sub_sq) < 0 : sqrt_val_sub = 1j * np.sqrt(-sqrt_term_sub_sq)
     elif abs(np.real(sqrt_term_sub_sq)) < 1e-12 : sqrt_val_sub = 1e-6 + 0j
     else : sqrt_val_sub = np.sqrt(sqrt_term_sub_sq)

     if pol == 'p':
          if abs(sqrt_val_sub) < 1e-9:
              etasub = np.inf + 0j
          else:
              etasub = n_sub_complex**2 / sqrt_val_sub
     else:
          etasub = sqrt_val_sub

     return etainc, etasub

def calcul_empilement(nH, nL, nSub, l0, emp_str, l_range, l_step, a_range, a_step, inc, n_inter, substrat_fini, lambda_monitoring):
    try:
        emp = [float(e) for e in emp_str.split(',') if e.strip()]
        if not emp:
             raise ValueError("Empilement invalide ou vide.")
    except ValueError as e:
         st.error(f"Erreur dans la définition de l'empilement : {e}")
         return None, None

    l, theta_inc_spectral = np.arange(l_range[0], l_range[1] + l_step, l_step), np.radians(inc)
    l_ang, theta_inc_ang = np.array([l0]), np.radians(np.arange(a_range[0], a_range[1] + a_step, a_step))

    ep = []
    for i, e in enumerate(emp):
        Ni = nH if i % 2 == 0 else nL
        n_real = np.real(Ni)
        if abs(n_real) < 1e-9:
             st.warning(f"Indice réel proche de zéro pour la couche {i+1}. L'épaisseur sera infinie ou indéfinie.")
             st.error(f"L'indice réel de la couche {i+1} (Matériau {'H' if i%2==0 else 'L'}) est proche de zéro.")
             return None, None
        ep.append(e * l0 / (4 * n_real))

    matrices_stockees = {}

    def calcul_RT_globale(longueurs_onde, angles, n_inc_medium=1.0):
        RT = np.zeros((len(longueurs_onde), len(angles), 4))

        for i_l, l_calc in enumerate(longueurs_onde):
            for i_a, theta in enumerate(angles):
                alpha = np.sin(theta)
                M_total_s = np.eye(2, dtype=complex)
                M_total_p = np.eye(2, dtype=complex)

                for i, e_ep in enumerate(ep):
                    Ni = nH if i % 2 == 0 else nL
                    key_s = (i, l_calc, theta, 's')
                    if key_s not in matrices_stockees:
                        matrices_stockees[key_s], _ = calcul_M('s', l_calc, theta, Ni, e_ep)
                    M_layer_s = matrices_stockees[key_s]
                    M_total_s = M_layer_s @ M_total_s

                    key_p = (i, l_calc, theta, 'p')
                    if key_p not in matrices_stockees:
                         matrices_stockees[key_p], _ = calcul_M('p', l_calc, theta, Ni, e_ep)
                    M_layer_p = matrices_stockees[key_p]
                    M_total_p = M_layer_p @ M_total_p

                etainc_s, etasub_s = calcul_etats('s', theta, n_inc_medium, nSub)
                etainc_p, etasub_p = calcul_etats('p', theta, n_inc_medium, nSub)

                Ms = M_total_s
                denom_s = (etainc_s * Ms[0, 0] + etasub_s * Ms[1, 1] + etainc_s * etasub_s * Ms[0, 1] + Ms[1, 0])
                if abs(denom_s) < 1e-12:
                    rs_infini = 1.0
                    ts_infini = 0.0
                else:
                    rs_infini = (etainc_s * Ms[0, 0] - etasub_s * Ms[1, 1] + etainc_s * etasub_s * Ms[0, 1] - Ms[1, 0]) / denom_s
                    ts_infini = 2 * etainc_s / denom_s

                Rs_infini = np.abs(rs_infini)**2
                if abs(np.real(etainc_s)) < 1e-12:
                     Ts_infini = 0.0
                else:
                     Ts_infini = (np.real(etasub_s) / np.real(etainc_s)) * np.abs(ts_infini)**2

                Mp = M_total_p
                denom_p = (etainc_p * Mp[0, 0] + etasub_p * Mp[1, 1] + etainc_p * etasub_p * Mp[0, 1] + Mp[1, 0])
                if abs(denom_p) < 1e-12:
                    rp_infini = 1.0
                    tp_infini = 0.0
                else:
                    rp_infini = (etainc_p * Mp[0, 0] - etasub_p * Mp[1, 1] + etainc_p * etasub_p * Mp[0, 1] - Mp[1, 0]) / denom_p
                    tp_infini = 2 * etainc_p / denom_p

                Rp_infini = np.abs(rp_infini)**2
                if abs(np.real(etainc_p)) < 1e-12:
                     Tp_infini = 0.0
                else:
                     Tp_infini = (np.real(etasub_p) / np.real(etainc_p)) * np.abs(tp_infini)**2

                if substrat_fini:
                    etainc_rev_s, etasub_rev_s = calcul_etats('s', theta, nSub, n_inc_medium)
                    etainc_rev_p, etasub_rev_p = calcul_etats('p', theta, nSub, n_inc_medium)

                    denom_bs = etainc_rev_s + etasub_rev_s
                    Rb_s = np.abs((etainc_rev_s - etasub_rev_s) / denom_bs)**2 if abs(denom_bs) > 1e-12 else 1.0

                    denom_bp = etainc_rev_p + etasub_rev_p
                    Rb_p = np.abs((etainc_rev_p - etasub_rev_p) / denom_bp)**2 if abs(denom_bp) > 1e-12 else 1.0

                    denom_Rs = 1 - Rs_infini * Rb_s
                    Rs = Rs_infini + (Ts_infini * Rb_s * Ts_infini / denom_Rs) if abs(denom_Rs) > 1e-12 else Rs_infini
                    Ts = (Ts_infini * (1 - Rb_s) / denom_Rs) if abs(denom_Rs) > 1e-12 else Ts_infini * (1 - Rb_s)

                    denom_Rp = 1 - Rp_infini * Rb_p
                    Rp = Rp_infini + (Tp_infini * Rb_p * Tp_infini / denom_Rp) if abs(denom_Rp) > 1e-12 else Rp_infini
                    Tp = (Tp_infini * (1 - Rb_p) / denom_Rp) if abs(denom_Rp) > 1e-12 else Tp_infini * (1 - Rb_p)

                else:
                    Rs, Ts, Rp, Tp = Rs_infini, Ts_infini, Rp_infini, Tp_infini

                RT[i_l, i_a, 0] = np.clip(np.nan_to_num(Rs), 0, 1)
                RT[i_l, i_a, 1] = np.clip(np.nan_to_num(Rp), 0, 1)
                RT[i_l, i_a, 2] = np.clip(np.nan_to_num(Ts), 0, 1)
                RT[i_l, i_a, 3] = np.clip(np.nan_to_num(Tp), 0, 1)

        return RT

    RT_s = calcul_RT_globale(l, [theta_inc_spectral])
    RT_a = calcul_RT_globale(l_ang, theta_inc_ang)

    transmissions = []
    transmissions_inter = []
    x_coords_inter = []
    ep_cum = np.cumsum(ep)
    n_inc_medium = 1.0

    theta_monitor = theta_inc_spectral
    etainc_s_nu, etasub_s_nu = calcul_etats('s', theta_monitor, n_inc_medium, nSub)
    etainc_p_nu, etasub_p_nu = calcul_etats('p', theta_monitor, n_inc_medium, nSub)

    denom_s_nu = etainc_s_nu + etasub_s_nu
    ts_nu = 2 * etainc_s_nu / denom_s_nu if abs(denom_s_nu) > 1e-12 else 0.0
    Ts_nu_infini = (np.real(etasub_s_nu) / np.real(etainc_s_nu)) * np.abs(ts_nu)**2 if abs(np.real(etainc_s_nu)) > 1e-12 else 0.0

    if substrat_fini:
         etainc_rev_s_nu, etasub_rev_s_nu = calcul_etats('s', theta_monitor, nSub, n_inc_medium)
         denom_bs_nu = etainc_rev_s_nu + etasub_rev_s_nu
         Rb_s_nu = np.abs((etainc_rev_s_nu - etasub_rev_s_nu) / denom_bs_nu)**2 if abs(denom_bs_nu) > 1e-12 else 1.0
         Rs_nu_infini = np.abs((etainc_s_nu - etasub_s_nu) / denom_s_nu)**2 if abs(denom_s_nu) > 1e-12 else 1.0
         denom_T_nu = 1 - Rs_nu_infini * Rb_s_nu
         T_substrat_nu = (Ts_nu_infini * (1 - Rb_s_nu) / denom_T_nu) if abs(denom_T_nu) > 1e-12 else Ts_nu_infini * (1 - Rb_s_nu)
    else:
         T_substrat_nu = Ts_nu_infini

    transmissions.append(np.clip(np.nan_to_num(T_substrat_nu), 0, 1))

    M_cumul_s = np.eye(2, dtype=complex)
    current_ep_cum = 0.0
    for i, e_layer in enumerate(ep):
        Ni = nH if i % 2 == 0 else nL
        for k in range(1, n_inter + 1):
             ep_partielle = e_layer * k / (n_inter + 1)
             M_segment_s, _ = calcul_M('s', lambda_monitoring, theta_monitor, Ni, ep_partielle)
             M_inter = M_segment_s @ M_cumul_s

             denom_s = (etainc_s_nu * M_inter[0, 0] + etasub_s_nu * M_inter[1, 1] + etainc_s_nu * etasub_s_nu * M_inter[0, 1] + M_inter[1, 0])
             ts_infini = 2 * etainc_s_nu / denom_s if abs(denom_s) > 1e-12 else 0.0
             Ts_infini = (np.real(etasub_s_nu) / np.real(etainc_s_nu)) * np.abs(ts_infini)**2 if abs(np.real(etainc_s_nu)) > 1e-12 else 0.0

             if substrat_fini:
                  rs_infini = (etainc_s_nu * M_inter[0, 0] - etasub_s_nu * M_inter[1, 1] + etainc_s_nu * etasub_s_nu * M_inter[0, 1] - M_inter[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
                  Rs_infini = np.abs(rs_infini)**2
                  denom_T = 1 - Rs_infini * Rb_s_nu
                  T_inter = (Ts_infini * (1 - Rb_s_nu) / denom_T) if abs(denom_T) > 1e-12 else Ts_infini * (1 - Rb_s_nu)
             else:
                  T_inter = Ts_infini

             transmissions_inter.append(np.clip(np.nan_to_num(T_inter), 0, 1))
             x_coords_inter.append(current_ep_cum + ep_partielle)

        M_layer_s, _ = calcul_M('s', lambda_monitoring, theta_monitor, Ni, e_layer)
        M_cumul_s = M_layer_s @ M_cumul_s
        current_ep_cum += e_layer

        denom_s = (etainc_s_nu * M_cumul_s[0, 0] + etasub_s_nu * M_cumul_s[1, 1] + etainc_s_nu * etasub_s_nu * M_cumul_s[0, 1] + M_cumul_s[1, 0])
        ts_infini = 2 * etainc_s_nu / denom_s if abs(denom_s) > 1e-12 else 0.0
        Ts_infini = (np.real(etasub_s_nu) / np.real(etainc_s_nu)) * np.abs(ts_infini)**2 if abs(np.real(etainc_s_nu)) > 1e-12 else 0.0

        if substrat_fini:
             rs_infini = (etainc_s_nu * M_cumul_s[0, 0] - etasub_s_nu * M_cumul_s[1, 1] + etainc_s_nu * etasub_s_nu * M_cumul_s[0, 1] - M_cumul_s[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
             Rs_infini = np.abs(rs_infini)**2
             denom_T = 1 - Rs_infini * Rb_s_nu
             T_fin_couche = (Ts_infini * (1 - Rb_s_nu) / denom_T) if abs(denom_T) > 1e-12 else Ts_infini * (1 - Rb_s_nu)
        else:
             T_fin_couche = Ts_infini

        transmissions.append(np.clip(np.nan_to_num(T_fin_couche), 0, 1))

    results = {
        'l': l, 'inc_spectral': np.array([inc]),
        'Rs_s': RT_s[:, 0, 0], 'Rp_s': RT_s[:, 0, 1], 'Ts_s': RT_s[:, 0, 2], 'Tp_s': RT_s[:, 0, 3],
        'l_a': l_ang, 'inc_a': np.degrees(theta_inc_ang),
        'Rs_a': RT_a[0, :, 0], 'Rp_a': RT_a[0, :, 1], 'Ts_a': RT_a[0, :, 2], 'Tp_a': RT_a[0, :, 3],
        'transmissions_interfaces': transmissions,
        'transmissions_intermediaires': transmissions_inter,
        'epaisseurs_intermediaires': x_coords_inter,
        'epaisseurs_interfaces': np.concatenate(([0.0], ep_cum)),
        'lambda_monitoring': lambda_monitoring,
        'n_inter': n_inter,
        'ep_layers': ep
    }

    return results, emp


def plot_spectral(res, params):
    fig, ax = plt.subplots(figsize=(7, 5))
    l_plt = res['l']
    inc = params['inc']

    ax.plot(l_plt, res['Rs_s'], label='Rs', linestyle='-', color='blue')
    ax.plot(l_plt, res['Rp_s'], label='Rp', linestyle='--', color='cyan')
    ax.plot(l_plt, res['Ts_s'], label='Ts', linestyle='-', color='red')
    ax.plot(l_plt, res['Tp_s'], label='Tp', linestyle='--', color='orange')

    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel('Reflectance / Transmittance')
    ax.set_title(f"Tracé spectral (incidence {inc:.1f}°)")
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.set_ylim(bottom=-0.05, top=1.05)
    if len(l_plt) > 1:
        ax.set_xlim(l_plt[0], l_plt[-1])
    ax.legend()
    plt.tight_layout()
    return fig

def plot_angular(res, params):
    fig, ax = plt.subplots(figsize=(7, 5))
    angles_deg = res['inc_a']
    l0_plt = res['l_a'][0]

    ax.plot(angles_deg, res['Rs_a'], label='Rs', linestyle='-', color='blue')
    ax.plot(angles_deg, res['Rp_a'], label='Rp', linestyle='--', color='cyan')
    ax.plot(angles_deg, res['Ts_a'], label='Ts', linestyle='-', color='red')
    ax.plot(angles_deg, res['Tp_a'], label='Tp', linestyle='--', color='orange')

    ax.set_xlabel("Angle d'incidence (degrés)")
    ax.set_ylabel('Reflectance / Transmittance')
    ax.set_title(f"Tracé angulaire (λ = {l0_plt:.0f} nm)")
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.set_ylim(bottom=-0.05, top=1.05)
    if len(angles_deg) > 1:
        ax.set_xlim(angles_deg[0], angles_deg[-1])
    ax.legend()
    plt.tight_layout()
    return fig

def plot_index_profile_and_monitoring(res, params, emp):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    nH_r = np.real(params['nH'])
    nL_r = np.real(params['nL'])
    nSub = np.real(params['nSub'])
    lambda_monitoring = res['lambda_monitoring']
    n_inter = res['n_inter']
    ep_layers = res['ep_layers']

    indices_reels = [nH_r if i % 2 == 0 else nL_r for i in range(len(emp))]
    ep_cum = np.cumsum(ep_layers)

    last_ep_cum = ep_cum[-1] if len(ep_cum) > 0 else 0
    x_coords_indice = np.concatenate(([-50, 0], ep_cum, [last_ep_cum + 51]))
    y_coords_indice = np.concatenate(([nSub, nSub], indices_reels, [1]))

    ax1.plot(x_coords_indice, y_coords_indice, drawstyle='steps-post', label='n réel', color='green')
    ax1.set_xlabel('Epaisseur cumulée (nm)')
    ax1.set_ylabel('Partie réelle de l\'indice (n)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_title('Profil d\'indice et Monitoring')
    ax1.grid(which='major', axis='x', color='grey', linestyle='-', linewidth=0.7)
    ax1.grid(which='minor', axis='x', color='lightgrey', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()

    min_n = min(1.0, nSub, nH_r, nL_r)
    max_n = max(1.0, nSub, nH_r, nL_r)
    ax1.set_ylim(min_n - 0.1, max_n + 0.1)

    ax1.text(-25, (ax1.get_ylim()[0] + ax1.get_ylim()[1])/2 , "SUBSTRAT", ha='center', va='center', fontsize=8, rotation=90, color='gray')
    ax1.text(last_ep_cum + 25, (ax1.get_ylim()[0] + ax1.get_ylim()[1])/2, "AIR", ha='center', va='center', fontsize=8, rotation=90, color='gray')

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'T Monitoring (λ = {lambda_monitoring:.0f} nm)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.05, 1.05)

    x_monitoring_all = np.concatenate((res['epaisseurs_interfaces'], res['epaisseurs_intermediaires']))
    y_monitoring_all = np.concatenate((res['transmissions_interfaces'], res['transmissions_intermediaires']))
    sorted_indices = np.argsort(x_monitoring_all)
    x_monitoring_sorted = x_monitoring_all[sorted_indices]
    y_monitoring_sorted = y_monitoring_all[sorted_indices]

    x_smooth_start = -50
    x_smooth_end = last_ep_cum + 50
    x_monitoring_extended = np.concatenate(([x_smooth_start], x_monitoring_sorted, [x_smooth_end]))
    y_monitoring_extended = np.concatenate(([y_monitoring_sorted[0] if len(y_monitoring_sorted)>0 else 0], y_monitoring_sorted, [y_monitoring_sorted[-1] if len(y_monitoring_sorted)>0 else 0]))

    if len(x_monitoring_extended) > 3:
        num_smooth_points = max(100, len(x_monitoring_extended) * 5)
        x_coords_smooth = np.linspace(x_smooth_start, x_smooth_end, num_smooth_points)
        try:
             spl = make_interp_spline(x_monitoring_extended, y_monitoring_extended, k=3)
             y_coords_smooth = spl(x_coords_smooth)
             y_coords_smooth = np.clip(y_coords_smooth, 0, 1)
             line_transmittance, = ax2.plot(x_coords_smooth, y_coords_smooth, 'r-', linewidth=1.5, label=f'T @ {lambda_monitoring:.0f} nm')
        except Exception as e_spline:
             st.warning(f"Impossible de générer la spline de monitoring: {e_spline}. Tracé points seulement.")
             line_transmittance, = ax2.plot(x_monitoring_sorted, y_monitoring_sorted, 'r-', linewidth=1.5, label=f'T @ {lambda_monitoring:.0f} nm')

    else:
         line_transmittance, = ax2.plot(x_monitoring_extended, y_monitoring_extended, 'r-', linewidth=1.5, label=f'T @ {lambda_monitoring:.0f} nm')

    ax2.plot(x_monitoring_sorted, y_monitoring_sorted, 'ro', markersize=3, label='Points calculés')

    ax1.set_xlim(x_smooth_start, x_smooth_end)
    ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.tight_layout()
    return fig

def plot_stack_bars(res, params, emp):
    fig, ax = plt.subplots(figsize=(7, max(3, len(emp)*0.4)))

    nH_complex = params['nH']
    nL_complex = params['nL']
    ep_layers = res['ep_layers']

    indices_complex = [nH_complex if i % 2 == 0 else nL_complex for i in range(len(emp))]

    ep_layers_rev = np.array(ep_layers)[::-1]
    indices_complex_rev = indices_complex[::-1]
    couche_labels = [f"Couche {len(emp) - i}" for i in range(len(emp))]

    colors = ['lightblue' if np.real(n) == np.real(nH_complex) else 'lightcoral' for n in indices_complex_rev]

    bars = ax.barh(range(len(emp)), ep_layers_rev, align='center', color=colors, edgecolor='black')

    ax.set_yticks(range(len(emp)))
    ax.set_yticklabels(couche_labels, fontsize=9)
    ax.set_xlabel('Epaisseur (nm)')
    ax.set_title('Empilement (Substrat en bas, Couche 1 en bas)')

    fontsize = max(7, 10 - len(emp) // 4)
    for i, (bar, e, ind) in enumerate(zip(bars, ep_layers_rev, indices_complex_rev)):
        label_txt = f"{e:.1f} nm\nn={np.real(ind):.2f}{np.imag(ind):+.3f}j"
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, label_txt,
                va='center', ha='center', color='black', fontsize=fontsize)

    ax.axhline(-0.5, color='gray', linestyle='--', linewidth=2)
    ax.text(np.mean(ax.get_xlim()) if len(ep_layers_rev)>0 else 0.5, -0.6, 'Substrat', va='top', ha='center', color='gray', fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_rs_infini_complexe(res, params, emp):
    fig, ax = plt.subplots(figsize=(6, 6))

    lambda_monitoring = res['lambda_monitoring']
    theta_monitor = np.radians(params['inc'])
    nH = params['nH']
    nL = params['nL']
    nSub = params['nSub']
    n_inter = res['n_inter']
    ep_layers = res['ep_layers']
    substrat_fini = params['substrat_fini']
    n_inc_medium = 1.0

    etainc_s, etasub_s = calcul_etats('s', theta_monitor, n_inc_medium, nSub)

    rs_infinis_s = []
    epaisseurs_trace = []

    epaisseurs_trace.append(0.0)
    denom_s_nu = etainc_s + etasub_s
    rs_nu = (etainc_s - etasub_s) / denom_s_nu if abs(denom_s_nu) > 1e-12 else 1.0
    rs_infinis_s.append(rs_nu)

    M_cumul_s = np.eye(2, dtype=complex)
    current_ep_cum = 0.0
    for i, e_layer in enumerate(ep_layers):
        Ni = nH if i % 2 == 0 else nL
        color = 'blue' if i % 2 == 0 else 'red'
        for k in range(1, n_inter + 1):
            ep_partielle = e_layer * k / (n_inter + 1)
            M_segment_s, _ = calcul_M('s', lambda_monitoring, theta_monitor, Ni, ep_partielle)
            M_inter = M_segment_s @ M_cumul_s

            denom_s = (etainc_s * M_inter[0, 0] + etasub_s * M_inter[1, 1] + etainc_s * etasub_s * M_inter[0, 1] + M_inter[1, 0])
            rs_infini = (etainc_s * M_inter[0, 0] - etasub_s * M_inter[1, 1] + etainc_s * etasub_s * M_inter[0, 1] - M_inter[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
            rs_infinis_s.append(rs_infini)
            ep_cum_inter = current_ep_cum + ep_partielle
            epaisseurs_trace.append(ep_cum_inter)
            ax.plot(np.real(rs_infini), np.imag(rs_infini), marker='o', linestyle='none', markersize=2, color=color, alpha=0.6)

        M_layer_s, _ = calcul_M('s', lambda_monitoring, theta_monitor, Ni, e_layer)
        M_cumul_s = M_layer_s @ M_cumul_s
        current_ep_cum += e_layer

        denom_s = (etainc_s * M_cumul_s[0, 0] + etasub_s * M_cumul_s[1, 1] + etainc_s * etasub_s * M_cumul_s[0, 1] + M_cumul_s[1, 0])
        rs_infini = (etainc_s * M_cumul_s[0, 0] - etasub_s * M_cumul_s[1, 1] + etainc_s * etasub_s * M_cumul_s[0, 1] - M_cumul_s[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
        rs_infinis_s.append(rs_infini)
        epaisseurs_trace.append(current_ep_cum)
        ax.plot(np.real(rs_infini), np.imag(rs_infini), marker='o', linestyle='none', markersize=5, color=color, markeredgecolor='black')
        ax.annotate(f"{i+1}", (np.real(rs_infini), np.imag(rs_infini)),
                    textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)

    epaisseurs_trace = np.array(epaisseurs_trace)
    rs_infinis_s = np.array(rs_infinis_s)
    sort_indices = np.argsort(epaisseurs_trace)
    epaisseurs_sorted = epaisseurs_trace[sort_indices]
    rs_sorted_real = np.real(rs_infinis_s[sort_indices])
    rs_sorted_imag = np.imag(rs_infinis_s[sort_indices])

    if len(epaisseurs_sorted) > 3 :
        try:
            spl_real = make_interp_spline(epaisseurs_sorted, rs_sorted_real, k=3)
            spl_imag = make_interp_spline(epaisseurs_sorted, rs_sorted_imag, k=3)
            ep_smooth = np.linspace(epaisseurs_sorted.min(), epaisseurs_sorted.max(), 300)
            rs_smooth_real = spl_real(ep_smooth)
            rs_smooth_imag = spl_imag(ep_smooth)
            ax.plot(rs_smooth_real, rs_smooth_imag, '-', color='black', linewidth=0.8, alpha=0.7, zorder=-1)
        except Exception as e_spline_complex:
             st.warning(f"Impossible de lisser la courbe complexe: {e_spline_complex}. Tracé linéaire.")
             ax.plot(rs_sorted_real, rs_sorted_imag, '-', color='black', linewidth=0.8, alpha=0.7, zorder=-1)
    else:
        ax.plot(rs_sorted_real, rs_sorted_imag, '-', color='black', linewidth=0.8, alpha=0.7, zorder=-1)

    ax.set_xlabel('Re(rs)')
    ax.set_ylabel('Im(rs)')
    ax.set_title(f'Plan complexe de rs (Pol S, λ={lambda_monitoring:.0f} nm, θ={params["inc"]:.1f}°)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    max_abs_r = np.max(np.abs(rs_infinis_s)) if len(rs_infinis_s) > 0 else 1.0
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]), max_abs_r * 1.1) if len(rs_infinis_s) > 0 else 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    return fig

def create_excel_output(res, params, emp):
    output = io.BytesIO()
    dfs = {}

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        params_dict_for_export = params.copy()
        params_dict_for_export['nH'] = f"{np.real(params['nH']):.4f}{np.imag(params['nH']):+.4f}j"
        params_dict_for_export['nL'] = f"{np.real(params['nL']):.4f}{np.imag(params['nL']):+.4f}j"
        dfs['Paramètres'] = pd.DataFrame.from_dict(params_dict_for_export, orient='index', columns=['Valeur'])
        dfs['Paramètres'].loc['Empilement String'] = params['emp_str']
        dfs['Paramètres'].to_excel(writer, sheet_name='Paramètres')

        dfs['Données Spectrales'] = pd.DataFrame({
            'Longueur d\'onde (nm)': res['l'],
            'Rs': res['Rs_s'], 'Rp': res['Rp_s'],
            'Ts': res['Ts_s'], 'Tp': res['Tp_s']
        })
        dfs['Données Spectrales'].to_excel(writer, sheet_name='Données Spectrales', index=False)

        dfs['Données Angulaires'] = pd.DataFrame({
            'Angle (°)': res['inc_a'],
            'Rs': res['Rs_a'], 'Rp': res['Rp_a'],
            'Ts': res['Ts_a'], 'Tp': res['Tp_a']
        })
        dfs['Données Angulaires'].to_excel(writer, sheet_name='Données Angulaires', index=False)

        x_monitoring_all = np.concatenate((res['epaisseurs_interfaces'], res['epaisseurs_intermediaires']))
        y_monitoring_all = np.concatenate((res['transmissions_interfaces'], res['transmissions_intermediaires']))
        sorted_indices = np.argsort(x_monitoring_all)
        dfs['Monitoring T'] = pd.DataFrame({
            f'Epaisseur cumulée (nm) @ {res["lambda_monitoring"]:.0f}nm': x_monitoring_all[sorted_indices],
            f'Transmission @ {res["lambda_monitoring"]:.0f}nm': y_monitoring_all[sorted_indices]
        })
        dfs['Monitoring T'].to_excel(writer, sheet_name='Monitoring T', index=False)

        dfs['Couches Details'] = pd.DataFrame({
             'Couche #': [i + 1 for i in range(len(emp))],
             'Type': ['H' if i % 2 == 0 else 'L' for i in range(len(emp))],
             'Indice Complexe': [f"{np.real(params['nH']):.4f}{np.imag(params['nH']):+.4f}j" if i % 2 == 0 else f"{np.real(params['nL']):.4f}{np.imag(params['nL']):+.4f}j" for i in range(len(emp))],
             'Epaisseur (nm)': res['ep_layers']
        })
        dfs['Couches Details'].to_excel(writer, sheet_name='Couches Details', index=False)

        workbook = writer.book
        for sheet_name, df_current in dfs.items():
            worksheet = writer.sheets[sheet_name]
            is_params_sheet = (sheet_name == 'Paramètres')

            for i, col in enumerate(df_current.columns):
                if is_params_sheet and i == 0:
                     idx_len = df_current.index.astype(str).map(len).max()
                     col_len = df_current[col].astype(str).map(len).max()
                     header_len = max(len(str(col)), len(df_current.index.name) if df_current.index.name else 0)
                     data_len = max(idx_len if pd.notna(idx_len) else 0, col_len if pd.notna(col_len) else 0)
                else:
                    col_len = df_current[col].astype(str).map(len).max()
                    header_len = len(str(col))
                    data_len = col_len if pd.notna(col_len) else 0

                max_len = max(data_len, header_len) + 2
                col_idx_to_set = i + 1 if is_params_sheet else i
                worksheet.set_column(col_idx_to_set, col_idx_to_set, max_len)

            if is_params_sheet:
                 idx_name_len = len(df_current.index.name) if df_current.index.name else 0
                 idx_value_len = df_current.index.astype(str).map(len).max()
                 idx_width = max(idx_name_len, idx_value_len if pd.notna(idx_value_len) else 0) + 2
                 worksheet.set_column(0, 0, idx_width)

    output.seek(0)
    return output


st.set_page_config(page_title="Calcul Couches Minces", layout="wide")

default_params = {
    'nH_r': 2.25, 'nH_i': 0.0001,
    'nL_r': 1.48, 'nL_i': 0.0001,
    'nSub': 1.52,
    'l0': 550.0,
    'l_range_deb': 400.0, 'l_range_fin': 700.0, 'l_step': 1.0,
    'a_range_deb': 0.0, 'a_range_fin': 89.0, 'a_step': 1.0,
    'inc': 0.0,
    'n_inter': 30,
    'lambda_monitoring': 550.0,
    'emp_str': "1,1,1,1,1,2,1,1,1,1,1",
    'substrat_fini': False,
    'export_excel': False,
    'results': None,
    'emp_list': None,
    'fig_spectral': None,
    'fig_angular': None,
    'fig_profile_monitoring': None,
    'fig_stack': None,
    'fig_complex': None,
}

for key, value in default_params.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.sidebar.header("Paramètres de Simulation")

with st.sidebar.expander("Indices Optiques", expanded=True):
    st.session_state.nH_r = st.number_input("Matériau H (réel)", value=st.session_state.nH_r, step=0.01, format="%.4f")
    st.session_state.nH_i = st.number_input("Matériau H (imaginaire)", value=st.session_state.nH_i, step=0.0001, format="%.4f", min_value=0.0)
    st.session_state.nL_r = st.number_input("Matériau L (réel)", value=st.session_state.nL_r, step=0.01, format="%.4f")
    st.session_state.nL_i = st.number_input("Matériau L (imaginaire)", value=st.session_state.nL_i, step=0.0001, format="%.4f", min_value=0.0)
    st.session_state.nSub = st.number_input("Substrat (indice réel)", value=st.session_state.nSub, step=0.01, format="%.4f")

with st.sidebar.expander("Empilement et Géométrie", expanded=True):
    st.session_state.l0 = st.number_input("λ de centrage QWOT (nm)", value=st.session_state.l0, step=1.0, min_value=0.1)
    st.session_state.emp_str = st.text_input("Empilement (QWOT, ex: 1,1,2,1)", value=st.session_state.emp_str)
    try:
        num_layers = len([e for e in st.session_state.emp_str.split(',') if e.strip()])
        st.caption(f"Nombre de couches : {num_layers}")
    except:
        st.caption("Nombre de couches : Erreur de format")

    st.session_state.inc = st.number_input("Incidence (degrés)", value=st.session_state.inc, step=1.0, min_value=0.0, max_value=89.9)
    st.session_state.substrat_fini = st.checkbox("Substrat fini (réflexions incohérentes)", value=st.session_state.substrat_fini)

with st.sidebar.expander("Plages de Calcul", expanded=False):
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.session_state.l_range_deb = st.number_input("λ spectral début (nm)", value=st.session_state.l_range_deb, step=1.0)
    with col_l2:
        st.session_state.l_range_fin = st.number_input("λ spectral fin (nm)", value=st.session_state.l_range_fin, step=1.0)
    with col_l3:
        st.session_state.l_step = st.number_input("Pas λ (nm)", value=st.session_state.l_step, step=0.1, min_value=0.1)

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.session_state.a_range_deb = st.number_input("Angle début (°)", value=st.session_state.a_range_deb, step=1.0)
    with col_a2:
        st.session_state.a_range_fin = st.number_input("Angle fin (°)", value=st.session_state.a_range_fin, step=1.0)
    with col_a3:
        st.session_state.a_step = st.number_input("Pas Angle (°)", value=st.session_state.a_step, step=0.1, min_value=0.1)

with st.sidebar.expander("Monitoring", expanded=False):
     st.session_state.lambda_monitoring = st.number_input("λ monitoring (nm)", value=st.session_state.lambda_monitoring, step=1.0)
     st.session_state.n_inter = st.number_input("Pts par couche (monitoring)", value=st.session_state.n_inter, step=1, min_value=0)


st.sidebar.markdown("---")
run_calculation = st.sidebar.button("🚀 Lancer le Calcul", use_container_width=True, type="primary")
st.sidebar.markdown("---")
st.session_state.export_excel = st.sidebar.checkbox("Préparer l'export Excel", value=st.session_state.export_excel)


if run_calculation:
    st.session_state.results = None
    st.session_state.emp_list = None
    st.session_state.fig_spectral = None
    st.session_state.fig_angular = None
    st.session_state.fig_profile_monitoring = None
    st.session_state.fig_stack = None
    st.session_state.fig_complex = None

    valid_input = True
    if st.session_state.l_range_fin <= st.session_state.l_range_deb:
        st.sidebar.error("λ spectral fin <= début")
        valid_input = False
    if st.session_state.a_range_fin <= st.session_state.a_range_deb:
        st.sidebar.error("Angle fin <= début")
        valid_input = False
    if not st.session_state.emp_str.strip():
         st.sidebar.error("L'empilement ne peut pas être vide.")
         valid_input = False
    if st.session_state.l_step <= 0 or st.session_state.a_step <= 0:
         st.sidebar.error("Les pas spectraux et angulaires doivent être > 0.")
         valid_input = False


    if valid_input:
        nH_complex = st.session_state.nH_r + 1j * st.session_state.nH_i
        nL_complex = st.session_state.nL_r + 1j * st.session_state.nL_i

        params = {
            'nH': nH_complex, 'nL': nL_complex, 'nSub': st.session_state.nSub,
            'l0': st.session_state.l0, 'emp_str': st.session_state.emp_str,
            'l_range': (st.session_state.l_range_deb, st.session_state.l_range_fin),
            'l_step': st.session_state.l_step,
            'a_range': (st.session_state.a_range_deb, st.session_state.a_range_fin),
            'a_step': st.session_state.a_step,
            'inc': st.session_state.inc, 'n_inter': st.session_state.n_inter,
            'substrat_fini': st.session_state.substrat_fini,
            'lambda_monitoring': st.session_state.lambda_monitoring
        }

        with st.spinner("Calcul en cours..."):
             try:
                results_calc, emp_list_validated = calcul_empilement(**params)

                if results_calc is not None and emp_list_validated is not None:
                    st.session_state.results = results_calc
                    st.session_state.emp_list = emp_list_validated
                    st.success("Calcul terminé avec succès!")

                    st.session_state.fig_spectral = plot_spectral(results_calc, params)
                    st.session_state.fig_angular = plot_angular(results_calc, params)
                    st.session_state.fig_profile_monitoring = plot_index_profile_and_monitoring(results_calc, params, emp_list_validated)
                    st.session_state.fig_stack = plot_stack_bars(results_calc, params, emp_list_validated)

                else:
                     st.error("Le calcul a échoué (voir messages d'erreur ci-dessus).")
                     st.session_state.results = None
                     st.session_state.emp_list = None


             except Exception as e:
                 st.error(f"Une erreur est survenue pendant le calcul : {e}")
                 import traceback
                 st.error(traceback.format_exc())
                 st.session_state.results = None
                 st.session_state.emp_list = None


st.title("Résultats de la Simulation d'Empilement")

if st.session_state.results:
    results = st.session_state.results
    params_used = {
            'nH': st.session_state.nH_r + 1j * st.session_state.nH_i,
            'nL': st.session_state.nL_r + 1j * st.session_state.nL_i,
            'nSub': st.session_state.nSub,
            'inc': st.session_state.inc,
            'substrat_fini': st.session_state.substrat_fini,
            'emp_str': st.session_state.emp_str
    }
    emp_list = st.session_state.emp_list

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Spectral & Angulaire", "🔬 Profil & Monitoring", "🏗️ Structure Empilement", "🌀 Plan Complexe rs"])

    with tab1:
        if st.session_state.fig_spectral:
            st.pyplot(st.session_state.fig_spectral)
        else:
            st.warning("Le graphique spectral n'a pas été généré.")

        if st.session_state.fig_angular:
             st.pyplot(st.session_state.fig_angular)
        else:
             st.warning("Le graphique angulaire n'a pas été généré.")

    with tab2:
        if st.session_state.fig_profile_monitoring:
             st.pyplot(st.session_state.fig_profile_monitoring)
        else:
             st.warning("Le graphique Profil/Monitoring n'a pas été généré.")

    with tab3:
         if st.session_state.fig_stack:
             st.pyplot(st.session_state.fig_stack)
         else:
             st.warning("Le graphique de la structure n'a pas été généré.")

    with tab4:
         st.write(f"Trace le coefficient de réflexion `rs` dans le plan complexe pour λ = {st.session_state.lambda_monitoring} nm et incidence = {st.session_state.inc}°.")
         if st.button("Tracer rs dans le plan complexe"):
             if emp_list is not None:
                 try:
                      with st.spinner("Génération du tracé complexe..."):
                         st.session_state.fig_complex = plot_rs_infini_complexe(results, params_used, emp_list)
                         st.pyplot(st.session_state.fig_complex)
                 except Exception as e_complex:
                      st.error(f"Erreur lors du tracé complexe : {e_complex}")
             else:
                 st.warning("Veuillez d'abord lancer un calcul réussi.")

         elif st.session_state.fig_complex:
               st.pyplot(st.session_state.fig_complex)
         else:
               st.info("Cliquez sur le bouton pour générer le graphique (après un calcul réussi).")


    if st.session_state.export_excel:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export")
        if emp_list is not None:
            try:
                with st.spinner("Préparation du fichier Excel..."):
                     excel_data = create_excel_output(results, params_used, emp_list)
                     num_layers_export = len(emp_list) if emp_list else 0
                     now = datetime.datetime.now()
                     timestamp = now.strftime("%Y%m%d-%H%M%S")
                     excel_filename = f"Resultats_empilement_{num_layers_export}_couches_{timestamp}.xlsx"

                     st.sidebar.download_button(
                         label="📥 Télécharger les Résultats (Excel)",
                         data=excel_data,
                         file_name=excel_filename,
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         use_container_width=True
                     )
            except Exception as e_excel:
                st.sidebar.error(f"Erreur lors de la création du fichier Excel: {e_excel}")
        else:
             st.sidebar.warning("Veuillez d'abord lancer un calcul réussi pour exporter.")


elif run_calculation:
     st.warning("Le calcul n'a pas pu aboutir. Vérifiez les paramètres et les messages d'erreur.")
else:
    st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer le Calcul'.")

st.sidebar.markdown("---")
st.sidebar.caption("Adaptation Streamlit v1.3")
