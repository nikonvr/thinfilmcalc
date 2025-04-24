# -*- coding: utf-8 -*-
# (Imports remain the same)
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
import traceback

# --- Calculation Functions (Numba-jitted) ---
@numba.jit(nopython=True, cache=True)
def calculate_transfer_matrix(polarization, wavelength, incidence_rad, n_layer, layer_thickness):
    # ... (code unchanged) ...
    alpha = np.sin(incidence_rad)
    n_layer_complex = np.complex128(n_layer)
    sqrt_term_sq = n_layer_complex**2 - alpha**2
    if np.real(sqrt_term_sq) < -1e-12:
         sqrt_val = 1j * np.sqrt(-sqrt_term_sq)
    elif abs(np.real(sqrt_term_sq)) < 1e-12 and abs(np.imag(sqrt_term_sq)) < 1e-12:
          sqrt_val = 1e-6 + 0j
    else:
         sqrt_val = np.sqrt(sqrt_term_sq)
    if polarization == 'p':
         if abs(sqrt_val) < 1e-9:
             eta = np.inf + 0j
         else:
              eta = n_layer_complex**2 / sqrt_val
    else:
         eta = sqrt_val
    phi = (2 * np.pi / wavelength) * sqrt_val * layer_thickness
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    sin_phi_over_eta = 0j
    if abs(eta) > 1e-12:
        sin_phi_over_eta = (1j / eta) * sin_phi
    elif abs(sin_phi) > 1e-9:
         sin_phi_over_eta = np.inf + 0j
    m_layer = np.array([[cos_phi, sin_phi_over_eta],
                       [1j * eta * sin_phi, cos_phi]], dtype=np.complex128)
    return m_layer, eta

@numba.jit(nopython=True, cache=True)
def calculate_admittances(polarization, incidence_rad, n_inc, n_sub):
    # ... (code unchanged) ...
     alpha = np.sin(incidence_rad)
     n_inc_complex = np.complex128(n_inc)
     n_sub_complex = np.complex128(n_sub)
     eta_inc, eta_sub = 0j, 0j
     sqrt_term_inc_sq = n_inc_complex**2 - alpha**2
     if np.real(sqrt_term_inc_sq) < -1e-12 : sqrt_val_inc = 1j * np.sqrt(-sqrt_term_inc_sq)
     elif abs(np.real(sqrt_term_inc_sq)) < 1e-12 and abs(np.imag(sqrt_term_inc_sq)) < 1e-12: sqrt_val_inc = 1e-6 + 0j
     else: sqrt_val_inc = np.sqrt(sqrt_term_inc_sq)
     if polarization == 'p':
          if abs(sqrt_val_inc) < 1e-9:
              eta_inc = np.inf + 0j
          else:
              eta_inc = n_inc_complex**2 / sqrt_val_inc
     else:
          eta_inc = sqrt_val_inc
     sqrt_term_sub_sq = n_sub_complex**2 - alpha**2
     if np.real(sqrt_term_sub_sq) < -1e-12 : sqrt_val_sub = 1j * np.sqrt(-sqrt_term_sub_sq)
     elif abs(np.real(sqrt_term_sub_sq)) < 1e-12 and abs(np.imag(sqrt_term_sub_sq)) < 1e-12 : sqrt_val_sub = 1e-6 + 0j
     else : sqrt_val_sub = np.sqrt(sqrt_term_sub_sq)
     if polarization == 'p':
          if abs(sqrt_val_sub) < 1e-9:
              eta_sub = np.inf + 0j
          else:
              eta_sub = n_sub_complex**2 / sqrt_val_sub
     else:
          eta_sub = sqrt_val_sub
     return eta_inc, eta_sub

# --- Calculation Functions (Not Jitted) ---
def calculate_global_RT(wavelengths, angles_rad, nH, nL, nSub, physical_thicknesses, finite_substrate, incident_medium_index=1.0):
    # ... (code unchanged) ...
    RT = np.zeros((len(wavelengths), len(angles_rad), 4))
    matrix_cache = {}
    for i_wl, wl in enumerate(wavelengths):
        for i_ang, ang_rad in enumerate(angles_rad):
            M_total_s = np.eye(2, dtype=np.complex128)
            M_total_p = np.eye(2, dtype=np.complex128)
            for i_layer, thickness in enumerate(physical_thicknesses):
                Ni = nH if i_layer % 2 == 0 else nL
                key_s = (i_layer, wl, ang_rad, 's')
                if key_s not in matrix_cache:
                    matrix_cache[key_s], _ = calculate_transfer_matrix('s', wl, ang_rad, Ni, thickness)
                M_layer_s = matrix_cache[key_s]
                M_total_s = M_layer_s @ M_total_s
                key_p = (i_layer, wl, ang_rad, 'p')
                if key_p not in matrix_cache:
                    matrix_cache[key_p], _ = calculate_transfer_matrix('p', wl, ang_rad, Ni, thickness)
                M_layer_p = matrix_cache[key_p]
                M_total_p = M_layer_p @ M_total_p
            eta_inc_s, eta_sub_s = calculate_admittances('s', ang_rad, incident_medium_index, nSub)
            eta_inc_p, eta_sub_p = calculate_admittances('p', ang_rad, incident_medium_index, nSub)
            Ms = M_total_s
            denom_s = (eta_inc_s * Ms[0, 0] + eta_sub_s * Ms[1, 1] + eta_inc_s * eta_sub_s * Ms[0, 1] + Ms[1, 0])
            if abs(denom_s) < 1e-12:
                rs_inf = 1.0; ts_inf = 0.0
            else:
                rs_inf = (eta_inc_s * Ms[0, 0] - eta_sub_s * Ms[1, 1] + eta_inc_s * eta_sub_s * Ms[0, 1] - Ms[1, 0]) / denom_s
                ts_inf = 2 * eta_inc_s / denom_s
            Rs_inf = np.abs(rs_inf)**2
            if abs(np.real(eta_inc_s)) < 1e-12: Ts_inf = 0.0
            else: Ts_inf = np.real(eta_sub_s) / np.real(eta_inc_s) * np.abs(ts_inf)**2
            Mp = M_total_p
            denom_p = (eta_inc_p * Mp[0, 0] + eta_sub_p * Mp[1, 1] + eta_inc_p * eta_sub_p * Mp[0, 1] + Mp[1, 0])
            if abs(denom_p) < 1e-12:
                rp_inf = 1.0; tp_inf = 0.0
            else:
                rp_inf = (eta_inc_p * Mp[0, 0] - eta_sub_p * Mp[1, 1] + eta_inc_p * eta_sub_p * Mp[0, 1] - Mp[1, 0]) / denom_p
                tp_inf = 2 * eta_inc_p / denom_p
            Rp_inf = np.abs(rp_inf)**2
            if abs(np.real(eta_inc_p)) < 1e-12: Tp_inf = 0.0
            else: Tp_inf = np.real(eta_sub_p) / np.real(eta_inc_p) * np.abs(tp_inf)**2
            if finite_substrate:
                eta_inc_rev_s, eta_sub_rev_s = calculate_admittances('s', ang_rad, nSub, incident_medium_index)
                eta_inc_rev_p, eta_sub_rev_p = calculate_admittances('p', ang_rad, nSub, incident_medium_index)
                denom_bs = eta_inc_rev_s + eta_sub_rev_s
                Rb_s = np.abs((eta_inc_rev_s - eta_sub_rev_s) / denom_bs)**2 if abs(denom_bs) > 1e-12 else 1.0
                denom_bp = eta_inc_rev_p + eta_sub_rev_p
                Rb_p = np.abs((eta_inc_rev_p - eta_sub_rev_p) / denom_bp)**2 if abs(denom_bp) > 1e-12 else 1.0
                denom_Rs_corr = 1 - Rs_inf * Rb_s
                Rs = Rs_inf + (Ts_inf * Rb_s * Ts_inf / denom_Rs_corr) if abs(denom_Rs_corr) > 1e-12 else Rs_inf
                Ts = (Ts_inf * (1 - Rb_s) / denom_Rs_corr) if abs(denom_Rs_corr) > 1e-12 else Ts_inf * (1 - Rb_s)
                denom_Rp_corr = 1 - Rp_inf * Rb_p
                Rp = Rp_inf + (Tp_inf * Rb_p * Tp_inf / denom_Rp_corr) if abs(denom_Rp_corr) > 1e-12 else Rp_inf
                Tp = (Tp_inf * (1 - Rb_p) / denom_Rp_corr) if abs(denom_Rp_corr) > 1e-12 else Tp_inf * (1 - Rb_p)
            else:
                Rs, Ts, Rp, Tp = Rs_inf, Ts_inf, Rp_inf, Tp_inf
            RT[i_wl, i_ang, 0] = np.clip(np.nan_to_num(Rs), 0, 1)
            RT[i_wl, i_ang, 1] = np.clip(np.nan_to_num(Rp), 0, 1)
            RT[i_wl, i_ang, 2] = np.clip(np.nan_to_num(Ts), 0, 1)
            RT[i_wl, i_ang, 3] = np.clip(np.nan_to_num(Tp), 0, 1)
    return RT

def calculate_stack_properties(nH, nL, nSub, l0, stack_string, wl_range, wl_step, ang_range, ang_step, incidence_angle_deg, points_per_layer, finite_substrate, monitoring_wavelength):
    # ... (code unchanged) ...
    try:
        layer_multipliers = [float(e) for e in stack_string.split(',') if e.strip()]
        if not layer_multipliers:
             raise ValueError("Stack definition is empty or invalid.")
    except ValueError as e:
         st.error(f"Error parsing stack definition: {e}")
         return None, None
    wavelengths = np.arange(wl_range[0], wl_range[1] + wl_step, wl_step)
    incidence_rad_spectral = np.radians(incidence_angle_deg)
    wavelength_angular = np.array([l0])
    angles_rad_angular = np.radians(np.arange(ang_range[0], ang_range[1] + ang_step, ang_step))
    physical_thicknesses = []
    for i, multiplier in enumerate(layer_multipliers):
        Ni = nH if i % 2 == 0 else nL
        n_real = np.real(Ni)
        if abs(n_real) < 1e-9:
             st.warning(f"Real part of refractive index near zero for layer {i+1}. Thickness calculation unstable.")
             st.error(f"Real index of layer {i+1} (Material {'H' if i%2==0 else 'L'}) is near zero.")
             return None, None
        physical_thicknesses.append(multiplier * l0 / (4 * n_real))
    RT_spectral = calculate_global_RT(wavelengths, [incidence_rad_spectral], nH, nL, nSub, physical_thicknesses, finite_substrate)
    RT_angular = calculate_global_RT(wavelength_angular, angles_rad_angular, nH, nL, nSub, physical_thicknesses, finite_substrate)
    transmissions_at_interfaces = []
    transmissions_intermediate = []
    thicknesses_intermediate = []
    cumulative_thicknesses = np.cumsum(physical_thicknesses)
    incident_medium_index = 1.0
    monitoring_angle_rad = incidence_rad_spectral
    eta_inc_s_bare, eta_sub_s_bare = calculate_admittances('s', monitoring_angle_rad, incident_medium_index, nSub)
    denom_s_bare = eta_inc_s_bare + eta_sub_s_bare
    ts_bare = 2 * eta_inc_s_bare / denom_s_bare if abs(denom_s_bare) > 1e-12 else 0.0
    Ts_bare_inf = 0.0
    if abs(np.real(eta_inc_s_bare)) > 1e-12:
        Ts_bare_inf = np.real(eta_sub_s_bare) / np.real(eta_inc_s_bare) * np.abs(ts_bare)**2
    Rb_s_mon = 0.0
    if finite_substrate:
         eta_inc_rev_s_bare, eta_sub_rev_s_bare = calculate_admittances('s', monitoring_angle_rad, nSub, incident_medium_index)
         denom_bs_bare = eta_inc_rev_s_bare + eta_sub_rev_s_bare
         Rb_s_bare = np.abs((eta_inc_rev_s_bare - eta_sub_rev_s_bare) / denom_bs_bare)**2 if abs(denom_bs_bare) > 1e-12 else 1.0
         Rs_bare_inf = np.abs((eta_inc_s_bare - eta_sub_s_bare) / denom_s_bare)**2 if abs(denom_s_bare) > 1e-12 else 1.0
         denom_T_bare_corr = 1 - Rs_bare_inf * Rb_s_bare
         T_substrate_bare = (Ts_bare_inf * (1 - Rb_s_bare) / denom_T_bare_corr) if abs(denom_T_bare_corr) > 1e-12 else Ts_bare_inf * (1 - Rb_s_bare)
         Rb_s_mon = Rb_s_bare
    else:
         T_substrate_bare = Ts_bare_inf
    transmissions_at_interfaces.append(np.clip(np.nan_to_num(T_substrate_bare), 0, 1))
    M_cumulative_s = np.eye(2, dtype=np.complex128)
    current_cumulative_thickness = 0.0
    eta_inc_s_mon, eta_sub_s_mon = calculate_admittances('s', monitoring_angle_rad, incident_medium_index, nSub)
    for i_layer, layer_thickness in enumerate(physical_thicknesses):
        Ni = nH if i_layer % 2 == 0 else nL
        M_before_layer = M_cumulative_s.copy()
        for k in range(1, points_per_layer + 1):
             partial_thickness = layer_thickness * k / (points_per_layer + 1)
             M_segment_s, _ = calculate_transfer_matrix('s', monitoring_wavelength, monitoring_angle_rad, Ni, partial_thickness)
             M_intermediate = M_segment_s @ M_before_layer
             denom_s = (eta_inc_s_mon * M_intermediate[0, 0] + eta_sub_s_mon * M_intermediate[1, 1] + eta_inc_s_mon * eta_sub_s_mon * M_intermediate[0, 1] + M_intermediate[1, 0])
             ts_inf = 2 * eta_inc_s_mon / denom_s if abs(denom_s) > 1e-12 else 0.0
             Ts_inf = 0.0
             if abs(np.real(eta_inc_s_mon)) > 1e-12:
                Ts_inf = np.real(eta_sub_s_mon) / np.real(eta_inc_s_mon) * np.abs(ts_inf)**2
             if finite_substrate:
                  rs_inf = (eta_inc_s_mon * M_intermediate[0, 0] - eta_sub_s_mon * M_intermediate[1, 1] + eta_inc_s_mon * eta_sub_s_mon * M_intermediate[0, 1] - M_intermediate[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
                  Rs_inf = np.abs(rs_inf)**2
                  denom_T_corr = 1 - Rs_inf * Rb_s_mon
                  T_intermediate = (Ts_inf * (1 - Rb_s_mon) / denom_T_corr) if abs(denom_T_corr) > 1e-12 else Ts_inf * (1 - Rb_s_mon)
             else:
                  T_intermediate = Ts_inf
             transmissions_intermediate.append(np.clip(np.nan_to_num(T_intermediate), 0, 1))
             thicknesses_intermediate.append(current_cumulative_thickness + partial_thickness)
        M_layer_s, _ = calculate_transfer_matrix('s', monitoring_wavelength, monitoring_angle_rad, Ni, layer_thickness)
        M_cumulative_s = M_layer_s @ M_before_layer
        denom_s = (eta_inc_s_mon * M_cumulative_s[0, 0] + eta_sub_s_mon * M_cumulative_s[1, 1] + eta_inc_s_mon * eta_sub_s_mon * M_cumulative_s[0, 1] + M_cumulative_s[1, 0])
        ts_inf = 2 * eta_inc_s_mon / denom_s if abs(denom_s) > 1e-12 else 0.0
        Ts_inf = 0.0
        if abs(np.real(eta_inc_s_mon)) > 1e-12:
            Ts_inf = np.real(eta_sub_s_mon) / np.real(eta_inc_s_mon) * np.abs(ts_inf)**2
        if finite_substrate:
             rs_inf = (eta_inc_s_mon * M_cumulative_s[0, 0] - eta_sub_s_mon * M_cumulative_s[1, 1] + eta_inc_s_mon * eta_sub_s_mon * M_cumulative_s[0, 1] - M_cumulative_s[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
             Rs_inf = np.abs(rs_inf)**2
             denom_T_corr = 1 - Rs_inf * Rb_s_mon
             T_end_layer = (Ts_inf * (1 - Rb_s_mon) / denom_T_corr) if abs(denom_T_corr) > 1e-12 else Ts_inf * (1 - Rb_s_mon)
        else:
             T_end_layer = Ts_inf
        transmissions_at_interfaces.append(np.clip(np.nan_to_num(T_end_layer), 0, 1))
        current_cumulative_thickness += layer_thickness
    results = {
        'wavelengths': wavelengths, 'incidence_angle_spectral_deg': np.array([incidence_angle_deg]),
        'Rs_spectral': RT_spectral[:, 0, 0], 'Rp_spectral': RT_spectral[:, 0, 1],
        'Ts_spectral': RT_spectral[:, 0, 2], 'Tp_spectral': RT_spectral[:, 0, 3],
        'wavelength_angular': wavelength_angular, 'angles_deg_angular': np.degrees(angles_rad_angular),
        'Rs_angular': RT_angular[0, :, 0], 'Rp_angular': RT_angular[0, :, 1],
        'Ts_angular': RT_angular[0, :, 2], 'Tp_angular': RT_angular[0, :, 3],
        'transmissions_interfaces': transmissions_at_interfaces,
        'transmissions_intermediate': transmissions_intermediate,
        'thicknesses_intermediate': thicknesses_intermediate,
        'thicknesses_interfaces': np.concatenate(([0.0], cumulative_thicknesses)),
        'monitoring_wavelength': monitoring_wavelength,
        'points_per_layer': points_per_layer,
        'physical_thicknesses': physical_thicknesses
    }
    return results, layer_multipliers

# --- Plotting Functions ---
def plot_spectral_results(res, params):
    # ... (code unchanged) ...
    fig, ax = plt.subplots(figsize=(7, 5))
    wavelengths_plot = res['wavelengths']
    incidence_angle_deg = params['incidence_angle_deg']
    ax.plot(wavelengths_plot, res['Rs_spectral'], label='Rs', linestyle='-', color='blue')
    ax.plot(wavelengths_plot, res['Rp_spectral'], label='Rp', linestyle='--', color='cyan')
    ax.plot(wavelengths_plot, res['Ts_spectral'], label='Ts', linestyle='-', color='red')
    ax.plot(wavelengths_plot, res['Tp_spectral'], label='Tp', linestyle='--', color='orange')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel('Reflectance / Transmittance')
    ax.set_title(f"Spectral Scan (Incidence {incidence_angle_deg:.1f}°)")
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.set_ylim(bottom=-0.05, top=1.05)
    if len(wavelengths_plot) > 1:
        ax.set_xlim(wavelengths_plot[0], wavelengths_plot[-1])
    ax.legend()
    plt.tight_layout()
    return fig

def plot_angular_results(res, params):
    # ... (code unchanged) ...
    fig, ax = plt.subplots(figsize=(7, 5))
    angles_deg_plot = res['angles_deg_angular']
    l0_plot = res['wavelength_angular'][0]
    ax.plot(angles_deg_plot, res['Rs_angular'], label='Rs', linestyle='-', color='blue')
    ax.plot(angles_deg_plot, res['Rp_angular'], label='Rp', linestyle='--', color='cyan')
    ax.plot(angles_deg_plot, res['Ts_angular'], label='Ts', linestyle='-', color='red')
    ax.plot(angles_deg_plot, res['Tp_angular'], label='Tp', linestyle='--', color='orange')
    ax.set_xlabel("Incidence Angle (degrees)")
    ax.set_ylabel('Reflectance / Transmittance')
    ax.set_title(f"Angular Scan (λ = {l0_plot:.0f} nm)")
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.set_ylim(bottom=-0.05, top=1.05)
    if len(angles_deg_plot) > 1:
        ax.set_xlim(angles_deg_plot[0], angles_deg_plot[-1])
    ax.legend()
    plt.tight_layout()
    return fig

def plot_index_and_monitoring(res, params, layer_multipliers):
    # ... (code unchanged, using the latest corrections) ...
    fig, ax1 = plt.subplots(figsize=(7, 5))
    nH_r = np.real(params['nH'])
    nL_r = np.real(params['nL'])
    nSub = np.real(params['nSub'])
    monitoring_wavelength = res['monitoring_wavelength']
    physical_thicknesses = res['physical_thicknesses']
    real_indices = [nH_r if i % 2 == 0 else nL_r for i in range(len(layer_multipliers))]
    cumulative_thicknesses = np.cumsum(physical_thicknesses)
    last_cumulative_thickness = cumulative_thicknesses[-1] if len(cumulative_thicknesses) > 0 else 0
    if len(cumulative_thicknesses) > 0:
        x_coords_index = np.concatenate(([-50, 0], cumulative_thicknesses, [last_cumulative_thickness + 51]))
        y_coords_index = np.concatenate(([nSub], real_indices, [1, 1]))
    else:
        x_coords_index = np.array([-50, 0, 51])
        y_coords_index = np.array([nSub, 1, 1])
    ax1.plot(x_coords_index, y_coords_index, drawstyle='steps-post', label='n (real)', color='green')
    ax1.set_xlabel('Cumulative Thickness (nm)')
    ax1.set_ylabel('Real Part of Refractive Index (n)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_title('Index Profile & Monitoring Curve')
    ax1.grid(which='major', axis='x', color='grey', linestyle='-', linewidth=0.7)
    ax1.grid(which='minor', axis='x', color='lightgrey', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()
    min_n_val = min([1.0, nSub] + real_indices) if real_indices else min(1.0, nSub)
    max_n_val = max([1.0, nSub] + real_indices) if real_indices else max(1.0, nSub)
    ax1.set_ylim(min_n_val - 0.1, max_n_val + 0.1)
    ax1.text(-25, (ax1.get_ylim()[0] + ax1.get_ylim()[1])/2 , "SUBSTRATE", ha='center', va='center', fontsize=8, rotation=90, color='gray')
    ax1.text(last_cumulative_thickness + 25, (ax1.get_ylim()[0] + ax1.get_ylim()[1])/2, "AIR", ha='center', va='center', fontsize=8, rotation=90, color='gray')
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'T Monitoring (λ = {monitoring_wavelength:.0f} nm)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.05, 1.05)
    thicknesses_all_monitoring = np.concatenate((res['thicknesses_interfaces'], res['thicknesses_intermediate']))
    transmissions_all_monitoring = np.concatenate((res['transmissions_interfaces'], res['transmissions_intermediate']))
    if len(thicknesses_all_monitoring) > 0:
        sorted_indices = np.argsort(thicknesses_all_monitoring)
        thicknesses_sorted = thicknesses_all_monitoring[sorted_indices]
        transmissions_sorted = transmissions_all_monitoring[sorted_indices]
    else:
        thicknesses_sorted = np.array([0])
        transmissions_sorted = np.array([0])
    x_smooth_start = -50
    x_smooth_end = last_cumulative_thickness + 50
    ax2.plot(thicknesses_sorted, transmissions_sorted, 'ro', markersize=3, label='Calculated points')
    if len(thicknesses_sorted) > 3 and last_cumulative_thickness > 0:
        valid_spline_indices = np.where(thicknesses_sorted >= 0)[0]
        if len(valid_spline_indices) > 3:
            thickness_spline_in = thicknesses_sorted[valid_spline_indices]
            trans_spline_in = transmissions_sorted[valid_spline_indices]
            unique_indices = np.unique(thickness_spline_in, return_index=True)[1]
            thickness_spline_in = thickness_spline_in[np.sort(unique_indices)]
            trans_spline_in = trans_spline_in[np.sort(unique_indices)]
            if len(thickness_spline_in) > 3:
                num_smooth_points = max(100, len(thickness_spline_in) * 5)
                thickness_coords_smooth = np.linspace(0, last_cumulative_thickness, num_smooth_points)
                try:
                     spl = make_interp_spline(thickness_spline_in, trans_spline_in, k=3, bc_type='natural')
                     transmissions_coords_smooth = spl(thickness_coords_smooth)
                     transmissions_coords_smooth = np.clip(transmissions_coords_smooth, 0, 1)
                     ax2.plot(thickness_coords_smooth, transmissions_coords_smooth, 'r-', linewidth=1.5, label=f'T @ {monitoring_wavelength:.0f} nm (smoothed)')
                except Exception as e_spline:
                     st.warning(f"Spline failed: {e_spline}. Plotting linear.")
                     ax2.plot(thicknesses_sorted[valid_spline_indices], transmissions_sorted[valid_spline_indices], 'r-', linewidth=1.0, label=f'T @ {monitoring_wavelength:.0f} nm (linear)')
            else:
                 ax2.plot(thicknesses_sorted, transmissions_sorted, 'r-', linewidth=1.0, label=f'T @ {monitoring_wavelength:.0f} nm (linear)')
        else:
             ax2.plot(thicknesses_sorted, transmissions_sorted, 'r-', linewidth=1.0, label=f'T @ {monitoring_wavelength:.0f} nm (linear)')
    elif len(thicknesses_sorted) > 0:
         ax2.plot(thicknesses_sorted, transmissions_sorted, 'r-', linewidth=1.0, label=f'T @ {monitoring_wavelength:.0f} nm (linear)')
    if len(transmissions_sorted) > 0:
        t_start = transmissions_sorted[0]
        t_end = transmissions_sorted[-1]
        ax2.plot([x_smooth_start, 0], [t_start, t_start], 'r-', linewidth=1.0, alpha=0.7, label='_nolegend_')
        ax2.plot([last_cumulative_thickness, x_smooth_end], [t_end, t_end], 'r-', linewidth=1.0, alpha=0.7, label='_nolegend_')
    ax1.set_xlim(x_smooth_start, x_smooth_end)
    ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.tight_layout()
    return fig

def plot_stack_structure(res, params, layer_multipliers):
    # ... (code unchanged) ...
    fig, ax = plt.subplots(figsize=(7, max(3, len(layer_multipliers)*0.4)))
    nH_complex = params['nH']
    nL_complex = params['nL']
    physical_thicknesses = res['physical_thicknesses']
    complex_indices = [nH_complex if i % 2 == 0 else nL_complex for i in range(len(layer_multipliers))]
    thicknesses_reversed = np.array(physical_thicknesses)[::-1]
    indices_reversed = complex_indices[::-1]
    layer_labels = [f"Layer {len(layer_multipliers) - i}" for i in range(len(layer_multipliers))]
    colors = ['lightblue' if np.real(n) == np.real(nH_complex) else 'lightcoral' for n in indices_reversed]
    bars = ax.barh(range(len(layer_multipliers)), thicknesses_reversed, align='center', color=colors, edgecolor='black')
    ax.set_yticks(range(len(layer_multipliers)))
    ax.set_yticklabels(layer_labels, fontsize=9)
    ax.set_xlabel('Thickness (nm)')
    ax.set_title('Stack Structure (Substrate at bottom, Layer 1 at bottom)')
    fontsize = max(7, 10 - len(layer_multipliers) // 4)
    for i, (bar, thickness, index) in enumerate(zip(bars, thicknesses_reversed, indices_reversed)):
        label_txt = f"{thickness:.1f} nm\nn={np.real(index):.2f}{np.imag(index):+.3f}j"
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, label_txt,
                va='center', ha='center', color='black', fontsize=fontsize)
    ax.axhline(-0.5, color='gray', linestyle='--', linewidth=2)
    ax.text(np.mean(ax.get_xlim()) if len(thicknesses_reversed)>0 else 0.5, -0.6, 'Substrate', va='top', ha='center', color='gray', fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_complex_rs(res, params, layer_multipliers):
    # ... (code unchanged) ...
    fig, ax = plt.subplots(figsize=(6, 6))
    monitoring_wavelength = res['monitoring_wavelength']
    monitoring_angle_rad = np.radians(params['incidence_angle_deg'])
    nH = params['nH']
    nL = params['nL']
    nSub = params['nSub']
    points_per_layer = res['points_per_layer']
    physical_thicknesses = res['physical_thicknesses']
    finite_substrate = params['finite_substrate']
    incident_medium_index = 1.0
    eta_inc_s, eta_sub_s = calculate_admittances('s', monitoring_angle_rad, incident_medium_index, nSub)
    rs_values_complex = []
    thickness_trace = []
    thickness_trace.append(0.0)
    denom_s_bare = eta_inc_s + eta_sub_s
    rs_bare = (eta_inc_s - eta_sub_s) / denom_s_bare if abs(denom_s_bare) > 1e-12 else 1.0
    rs_values_complex.append(rs_bare)
    M_cumulative_s = np.eye(2, dtype=np.complex128)
    current_cumulative_thickness = 0.0
    for i_layer, layer_thickness in enumerate(physical_thicknesses):
        Ni = nH if i_layer % 2 == 0 else nL
        color = 'blue' if i_layer % 2 == 0 else 'red'
        M_before_layer = M_cumulative_s.copy()
        for k in range(1, points_per_layer + 1):
            partial_thickness = layer_thickness * k / (points_per_layer + 1)
            M_segment_s, _ = calculate_transfer_matrix('s', monitoring_wavelength, monitoring_angle_rad, Ni, partial_thickness)
            M_intermediate = M_segment_s @ M_before_layer
            denom_s = (eta_inc_s * M_intermediate[0, 0] + eta_sub_s * M_intermediate[1, 1] + eta_inc_s * eta_sub_s * M_intermediate[0, 1] + M_intermediate[1, 0])
            rs_intermediate = (eta_inc_s * M_intermediate[0, 0] - eta_sub_s * M_intermediate[1, 1] + eta_inc_s * eta_sub_s * M_intermediate[0, 1] - M_intermediate[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
            rs_values_complex.append(rs_intermediate)
            cumulative_intermediate_thickness = current_cumulative_thickness + partial_thickness
            thickness_trace.append(cumulative_intermediate_thickness)
            ax.plot(np.real(rs_intermediate), np.imag(rs_intermediate), marker='o', linestyle='none', markersize=2, color=color, alpha=0.6)
        M_layer_s, _ = calculate_transfer_matrix('s', monitoring_wavelength, monitoring_angle_rad, Ni, layer_thickness)
        M_cumulative_s = M_layer_s @ M_before_layer
        current_cumulative_thickness += layer_thickness
        denom_s = (eta_inc_s * M_cumulative_s[0, 0] + eta_sub_s * M_cumulative_s[1, 1] + eta_inc_s * eta_sub_s * M_cumulative_s[0, 1] + M_cumulative_s[1, 0])
        rs_end_layer = (eta_inc_s * M_cumulative_s[0, 0] - eta_sub_s * M_cumulative_s[1, 1] + eta_inc_s * eta_sub_s * M_cumulative_s[0, 1] - M_cumulative_s[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
        rs_values_complex.append(rs_end_layer)
        thickness_trace.append(current_cumulative_thickness)
        ax.plot(np.real(rs_end_layer), np.imag(rs_end_layer), marker='o', linestyle='none', markersize=5, color=color, markeredgecolor='black')
        ax.annotate(f"{i_layer+1}", (np.real(rs_end_layer), np.imag(rs_end_layer)),
                    textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
    thickness_trace = np.array(thickness_trace)
    rs_values_complex = np.array(rs_values_complex)
    sort_indices = np.argsort(thickness_trace)
    thicknesses_sorted = thickness_trace[sort_indices]
    rs_sorted_real = np.real(rs_values_complex[sort_indices])
    rs_sorted_imag = np.imag(rs_values_complex[sort_indices])
    if len(thicknesses_sorted) > 3 :
        try:
            spl_real = make_interp_spline(thicknesses_sorted, rs_sorted_real, k=3)
            spl_imag = make_interp_spline(thicknesses_sorted, rs_sorted_imag, k=3)
            thickness_smooth = np.linspace(thicknesses_sorted.min(), thicknesses_sorted.max(), 300)
            rs_smooth_real = spl_real(thickness_smooth)
            rs_smooth_imag = spl_imag(thickness_smooth)
            ax.plot(rs_smooth_real, rs_smooth_imag, '-', color='black', linewidth=0.8, alpha=0.7, zorder=-1)
        except Exception as e_spline_complex:
             st.warning(f"Could not smooth complex curve: {e_spline_complex}. Plotting linear segments.")
             ax.plot(rs_sorted_real, rs_sorted_imag, '-', color='black', linewidth=0.8, alpha=0.7, zorder=-1)
    else:
        ax.plot(rs_sorted_real, rs_sorted_imag, '-', color='black', linewidth=0.8, alpha=0.7, zorder=-1)
    ax.set_xlabel('Re(rs)')
    ax.set_ylabel('Im(rs)')
    ax.set_title(f'Complex Plane: rs (S-Pol, λ={monitoring_wavelength:.0f} nm, θ={params["incidence_angle_deg"]:.1f}°)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    max_abs_r_val = np.max(np.abs(rs_values_complex)) if len(rs_values_complex) > 0 else 1.0
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    lim_val = max(abs(current_xlim[0]), abs(current_xlim[1]), abs(current_ylim[0]), abs(current_ylim[1]), max_abs_r_val * 1.1) if len(rs_values_complex) > 0 else 1.1
    ax.set_xlim(-lim_val, lim_val)
    ax.set_ylim(-lim_val, lim_val)
    plt.tight_layout()
    return fig

def generate_excel_output(res, params, layer_multipliers):
    # ... (code unchanged) ...
    output = io.BytesIO()
    dfs_for_excel = {}
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        params_dict_export = params.copy()
        params_dict_export['nH'] = f"{np.real(params['nH']):.4f}{np.imag(params['nH']):+.4f}j"
        params_dict_export['nL'] = f"{np.real(params['nL']):.4f}{np.imag(params['nL']):+.4f}j"
        if 'wl_range' in params_dict_export: params_dict_export['wl_range'] = str(params_dict_export['wl_range'])
        if 'ang_range' in params_dict_export: params_dict_export['ang_range'] = str(params_dict_export['ang_range'])
        dfs_for_excel['Parameters'] = pd.DataFrame.from_dict(params_dict_export, orient='index', columns=['Value'])
        dfs_for_excel['Parameters'].loc['Stack String'] = params['stack_string']
        dfs_for_excel['Parameters'].to_excel(writer, sheet_name='Parameters')
        dfs_for_excel['Spectral Data'] = pd.DataFrame({
            'Wavelength (nm)': res['wavelengths'],
            'Rs': res['Rs_spectral'], 'Rp': res['Rp_spectral'],
            'Ts': res['Ts_spectral'], 'Tp': res['Tp_spectral']
        })
        dfs_for_excel['Spectral Data'].to_excel(writer, sheet_name='Spectral Data', index=False)
        dfs_for_excel['Angular Data'] = pd.DataFrame({
            'Angle (deg)': res['angles_deg_angular'],
            'Rs': res['Rs_angular'], 'Rp': res['Rp_angular'],
            'Ts': res['Ts_angular'], 'Tp': res['Tp_angular']
        })
        dfs_for_excel['Angular Data'].to_excel(writer, sheet_name='Angular Data', index=False)
        thicknesses_all_monitoring = np.concatenate((res['thicknesses_interfaces'], res['thicknesses_intermediate']))
        transmissions_all_monitoring = np.concatenate((res['transmissions_interfaces'], res['transmissions_intermediate']))
        sorted_indices = np.argsort(thicknesses_all_monitoring)
        dfs_for_excel['Monitoring T'] = pd.DataFrame({
            f'Cumul. Thickness (nm) @ {res["monitoring_wavelength"]:.0f}nm': thicknesses_all_monitoring[sorted_indices],
            f'Transmission @ {res["monitoring_wavelength"]:.0f}nm': transmissions_all_monitoring[sorted_indices]
        })
        dfs_for_excel['Monitoring T'].to_excel(writer, sheet_name='Monitoring T', index=False)
        dfs_for_excel['Layer Details'] = pd.DataFrame({
             'Layer #': [i + 1 for i in range(len(layer_multipliers))],
             'Type': ['H' if i % 2 == 0 else 'L' for i in range(len(layer_multipliers))],
             'Complex Index': [f"{np.real(params['nH']):.4f}{np.imag(params['nH']):+.4f}j" if i % 2 == 0 else f"{np.real(params['nL']):.4f}{np.imag(params['nL']):+.4f}j" for i in range(len(layer_multipliers))],
             'Thickness (nm)': res['physical_thicknesses']
        })
        dfs_for_excel['Layer Details'].to_excel(writer, sheet_name='Layer Details', index=False)
        workbook = writer.book
        for sheet_name, df_current in dfs_for_excel.items():
            worksheet = writer.sheets[sheet_name]
            is_params_sheet = (sheet_name == 'Parameters')
            for i, col in enumerate(df_current.columns):
                try:
                    if not df_current[col].empty:
                        col_len = df_current[col].astype(str).map(len).max()
                    else:
                        col_len = 0
                    header_len = len(str(col))
                    data_len = col_len if pd.notna(col_len) else 0
                    max_len = max(data_len, header_len) + 2
                    col_idx_to_set = i + (1 if is_params_sheet else 0)
                    worksheet.set_column(col_idx_to_set, col_idx_to_set, max_len)
                except Exception as e_col_width:
                     st.warning(f"Could not set width for col '{col}' in sheet '{sheet_name}': {e_col_width}")
            if is_params_sheet:
                 try:
                     idx_name_len = len(df_current.index.name) if df_current.index.name else 0
                     idx_value_len = df_current.index.astype(str).map(len).max()
                     idx_width = max(idx_name_len, idx_value_len if pd.notna(idx_value_len) else 0) + 2
                     worksheet.set_column(0, 0, idx_width)
                 except Exception as e_idx_width:
                     st.warning(f"Could not set width for index col in sheet '{sheet_name}': {e_idx_width}")
    output.seek(0)
    return output

st.set_page_config(page_title="Thin Film Calculator", layout="wide")

default_params_state = {
    'nH_r': 2.25, 'nH_i': 0.0001,
    'nL_r': 1.48, 'nL_i': 0.0001,
    'nSub': 1.52,
    'l0': 550.0,
    'wl_range_start': 400.0, 'wl_range_end': 700.0, 'wl_step': 1.0,
    'ang_range_start': 0.0, 'ang_range_end': 89.0, 'ang_step': 1.0,
    'incidence_angle_deg': 0.0,
    'points_per_layer': 30,
    'monitoring_wavelength': 550.0,
    'stack_string': "1,1,1,1,1,2,1,1,1,1,1",
    'finite_substrate': False,
    'export_excel': False,
    'results': None,
    'layer_multipliers_list': None,
    'fig_spectral': None,
    'fig_angular': None,
    'fig_profile_monitoring': None,
    'fig_stack': None,
    'fig_complex': None,
}

for key, value in default_params_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

help_text = """
**User Guide - Thin Film Stack Calculator**

**1. Goal:**
This application calculates the theoretical reflectance (R) and transmittance (T) of a multilayer thin film stack deposited on a known substrate using the transfer matrix method. It computes results over specified spectral and angular ranges.

**2. Workflow:**
   - Use the sidebar on the left to configure all input parameters.
   - Click the "Run Calculation" button.
   - View the resulting plots and data organized in the tabs in the main area.
   - Optionally, enable and click the "Download Results (Excel)" button in the sidebar.

**3. Parameters (Sidebar):**

   * **Optical Indices:**
        * `Material H (real/imag)`: Real and imaginary parts of the refractive index for the high-index material (nH). Imaginary part (k) relates to absorption.
        * `Material L (real/imag)`: Real and imaginary parts for the low-index material (nL).
        * `Substrate (real index)`: Real part of the substrate's refractive index. (Note: Current version assumes a non-absorbing substrate, k=0).

   * **Stack and Geometry:**
        * `QWOT Center λ (nm)`: The center wavelength ($\lambda_0$) used for calculating layer thicknesses based on Quarter-Wave Optical Thickness (QWOT) multipliers.
        * `Stack Definition (QWOT)`: Define the layer stack as a comma-separated list of multipliers relative to $\lambda_0/4$. Layers alternate starting with H (from substrate side):
            * Example: `1` -> H(1) -> $\lambda_0/(4 n_H)$
            * Example: `1,1` -> H(1), L(1)
            * Example: `1,2,1` -> H(1), L(2), H(1) -> H($\lambda_0/4n_H$), L($2\lambda_0/4n_L$), H($\lambda_0/4n_H$)
        * `Incidence Angle (degrees)`: The angle at which light strikes the stack (0° = normal incidence). Used for spectral/monitoring calculations.
        * `Finite Substrate`: Check this box to account for incoherent reflections from the back surface of the substrate (relevant for thick, transparent substrates). If unchecked, the substrate is treated as infinitely thick (no back reflections).

   * **Calculation Ranges:**
        * `Spectral λ Start/End/Step (nm)`: Defines the wavelength range and resolution for the spectral plots.
        * `Angle Start/End/Step (deg)`: Defines the angle range and resolution for the angular plots (calculated at the QWOT Center λ).

   * **Monitoring:**
        * `Monitoring λ (nm)`: The specific wavelength used to calculate the 'Transmission vs. Thickness' monitoring curve.
        * `Pts per Layer (monitoring)`: Number of intermediate points calculated *within* each layer for the monitoring curve. More points give a smoother curve but take longer. 0 means only calculate at interfaces.

**4. Actions (Sidebar):**

   * `Run Calculation`: Starts the simulation based on the current parameters. Results will appear in the main area.
   * `Prepare Excel Export`: Check this box *before* running the calculation if you want to download the results.
   * `Download Results (Excel)`: Appears if the above box was checked and the calculation was successful. Click to download an Excel file with parameters and detailed results.

**5. Results (Main Area Tabs):**

   * **📈 Spectral & Angular:**
        * Shows R and T vs. Wavelength (at the specified incidence angle) for both S (Rs, Ts) and P (Rp, Tp) polarizations.
        * Shows R and T vs. Incidence Angle (at the QWOT center wavelength) for S and P polarizations.
   * **🔬 Profile & Monitoring:**
        * Shows the refractive index profile (real part 'n') as a function of cumulative physical thickness from the substrate.
        * Shows the calculated Transmission vs. cumulative physical thickness at the specified 'Monitoring λ'. Includes calculated points and an optional smoothed spline curve. The curve is held constant before thickness=0 and after the final thickness.
   * **🏗️ Stack Structure:**
        * Displays a bar chart visualizing the stack layer sequence (Layer 1 is at the bottom, near the substrate), material type (H/L based color), physical thickness (nm), and complex refractive index for each layer.
   * **🌀 Complex rs Plane:**
        * Click the "Plot rs..." button to generate a plot showing the evolution of the complex reflection coefficient `rs` (for S-polarization) as layers are deposited. This is calculated at the 'Monitoring λ' and specified incidence angle. Points mark intermediate steps and layer interfaces.

**6. Troubleshooting:**

   * **Errors on Run:** Check parameter sanity (e.g., End > Start for ranges, non-zero steps, valid stack string format, non-zero real indices). Look for specific error messages.
   * **Slow Calculation:** Reduce the number of spectral/angular points (increase step size), decrease 'Pts per Layer' for monitoring, or simulate simpler stacks. Numba optimization helps but large calculations still take time. The *first* run after code changes might be slower due to Numba compilation.
   * **Excel Export Issues:** Ensure calculation completed successfully and the checkbox was ticked *before* running. Check for warnings during Excel file preparation.
   * **Plot Issues:** If plots look incorrect, double-check input parameters (indices, stack definition) and calculation ranges. Ensure the monitoring wavelength is within the spectral range if comparing results.

**7. Contact:**
For further assistance, questions, or bug reports, please contact Fabien Lemarchand at: **fabien.lemarchand@gmail.com**
"""


st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("Optical Indices", expanded=True):
    st.number_input("Material H (real)", value=st.session_state.nH_r, step=0.01, format="%.4f", key="nH_r")
    st.number_input("Material H (imag)", value=st.session_state.nH_i, step=0.0001, format="%.4f", min_value=0.0, key="nH_i")
    st.number_input("Material L (real)", value=st.session_state.nL_r, step=0.01, format="%.4f", key="nL_r")
    st.number_input("Material L (imag)", value=st.session_state.nL_i, step=0.0001, format="%.4f", min_value=0.0, key="nL_i")
    st.number_input("Substrate (real index)", value=st.session_state.nSub, step=0.01, format="%.4f", key="nSub")

with st.sidebar.expander("Stack and Geometry", expanded=True):
    st.number_input("QWOT Center λ (nm)", value=st.session_state.l0, step=1.0, min_value=0.1, key="l0")
    st.text_input("Stack Definition (QWOT, e.g., 1,1,2,1)", value=st.session_state.stack_string, key="stack_string")
    try:
        num_layers_disp = len([e for e in st.session_state.stack_string.split(',') if e.strip()])
        st.caption(f"Number of layers: {num_layers_disp}")
    except:
        st.caption("Layer count: Format error")

    st.number_input("Incidence Angle (degrees)", value=st.session_state.incidence_angle_deg, step=1.0, min_value=0.0, max_value=89.9, key="incidence_angle_deg")
    st.checkbox("Finite Substrate (incoherent reflections)", value=st.session_state.finite_substrate, key="finite_substrate")

with st.sidebar.expander("Calculation Ranges", expanded=False):
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.number_input("Spectral λ Start (nm)", value=st.session_state.wl_range_start, step=1.0, key="wl_start")
    with col_l2:
        st.number_input("Spectral λ End (nm)", value=st.session_state.wl_range_end, step=1.0, key="wl_end")
    with col_l3:
        st.number_input("λ Step (nm)", value=st.session_state.wl_step, step=0.1, min_value=0.01, key="wl_step")

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.number_input("Angle Start (deg)", value=st.session_state.ang_range_start, step=1.0, key="ang_start")
    with col_a2:
        st.number_input("Angle End (deg)", value=st.session_state.ang_range_end, step=1.0, key="ang_end")
    with col_a3:
        st.number_input("Angle Step (deg)", value=st.session_state.ang_step, step=0.1, min_value=0.01, key="ang_step")

with st.sidebar.expander("Monitoring", expanded=False):
     st.number_input("Monitoring λ (nm)", value=st.session_state.monitoring_wavelength, step=1.0, key="mon_wl")
     st.number_input("Pts per Layer (monitoring)", value=st.session_state.points_per_layer, step=1, min_value=0, key="mon_pts")


st.sidebar.markdown("---")
run_calculation = st.sidebar.button("🚀 Run Calculation", use_container_width=True, type="primary")
st.sidebar.markdown("---")
st.checkbox("Prepare Excel Export", value=st.session_state.export_excel, key="export_cb")

with st.sidebar.expander("Help / User Guide"):
     st.markdown(help_text)


if run_calculation:
    st.session_state.results = None
    st.session_state.layer_multipliers_list = None
    st.session_state.fig_spectral = None
    st.session_state.fig_angular = None
    st.session_state.fig_profile_monitoring = None
    st.session_state.fig_stack = None
    st.session_state.fig_complex = None

    valid_input = True
    if st.session_state.wl_range_end <= st.session_state.wl_range_start:
        st.sidebar.error("Spectral λ End <= Start")
        valid_input = False
    if st.session_state.ang_range_end <= st.session_state.ang_range_start:
        st.sidebar.error("Angle End <= Start")
        valid_input = False
    if not st.session_state.stack_string.strip():
         st.sidebar.error("Stack definition cannot be empty.")
         valid_input = False
    if st.session_state.wl_step <= 0 or st.session_state.ang_step <= 0:
         st.sidebar.error("Spectral and Angular steps must be > 0.")
         valid_input = False
    if st.session_state.l0 <= 0:
         st.sidebar.error("QWOT Center λ must be > 0.")
         valid_input = False


    if valid_input:
        nH_complex = st.session_state.nH_r + 1j * st.session_state.nH_i
        nL_complex = st.session_state.nL_r + 1j * st.session_state.nL_i

        calc_params = {
            'nH': nH_complex, 'nL': nL_complex, 'nSub': st.session_state.nSub,
            'l0': st.session_state.l0, 'stack_string': st.session_state.stack_string,
            'wl_range': (st.session_state.wl_range_start, st.session_state.wl_range_end),
            'wl_step': st.session_state.wl_step,
            'ang_range': (st.session_state.ang_range_start, st.session_state.ang_range_end),
            'ang_step': st.session_state.ang_step,
            'incidence_angle_deg': st.session_state.incidence_angle_deg,
            'points_per_layer': st.session_state.points_per_layer,
            'finite_substrate': st.session_state.finite_substrate,
            'monitoring_wavelength': st.session_state.monitoring_wavelength
        }

        with st.spinner("Calculation in progress..."):
             try:
                results_calc, layer_multipliers_validated = calculate_stack_properties(**calc_params)

                if results_calc is not None and layer_multipliers_validated is not None:
                    st.session_state.results = results_calc
                    st.session_state.layer_multipliers_list = layer_multipliers_validated
                    st.success("Calculation successful!")

                    st.session_state.fig_spectral = plot_spectral_results(results_calc, calc_params)
                    st.session_state.fig_angular = plot_angular_results(results_calc, calc_params)
                    st.session_state.fig_profile_monitoring = plot_index_and_monitoring(results_calc, calc_params, layer_multipliers_validated)
                    st.session_state.fig_stack = plot_stack_structure(results_calc, calc_params, layer_multipliers_validated)

                else:
                     st.session_state.results = None
                     st.session_state.layer_multipliers_list = None


             except Exception as e:
                 st.error(f"An error occurred during calculation: {e}")
                 st.error(traceback.format_exc())
                 st.session_state.results = None
                 st.session_state.layer_multipliers_list = None


st.title("Thin Film Stack Simulation Results")

if st.session_state.results:
    results = st.session_state.results
    params_used_for_plots = {
            'nH': st.session_state.nH_r + 1j * st.session_state.nH_i,
            'nL': st.session_state.nL_r + 1j * st.session_state.nL_i,
            'nSub': st.session_state.nSub,
            'incidence_angle_deg': st.session_state.incidence_angle_deg,
            'finite_substrate': st.session_state.finite_substrate,
            'stack_string': st.session_state.stack_string
    }
    layer_multipliers_list = st.session_state.layer_multipliers_list

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Spectral & Angular", "🔬 Profile & Monitoring", "🏗️ Stack Structure", "🌀 Complex rs Plane"])

    with tab1:
        st.subheader("Spectral Response")
        if st.session_state.fig_spectral:
            st.pyplot(st.session_state.fig_spectral)
        else:
            st.warning("Spectral plot could not be generated.")

        st.subheader("Angular Response")
        if st.session_state.fig_angular:
             st.pyplot(st.session_state.fig_angular)
        else:
             st.warning("Angular plot could not be generated.")

    with tab2:
        st.subheader("Index Profile & Monitoring Curve")
        if st.session_state.fig_profile_monitoring:
             st.pyplot(st.session_state.fig_profile_monitoring)
        else:
             st.warning("Profile & Monitoring plot could not be generated.")

    with tab3:
         st.subheader("Stack Visualization")
         if st.session_state.fig_stack:
             st.pyplot(st.session_state.fig_stack)
         else:
             st.warning("Stack structure plot could not be generated.")

    with tab4:
         st.subheader("Complex Reflection Coefficient (rs)")
         st.write(f"Traces the S-polarization reflection coefficient `rs` in the complex plane during deposition.")
         st.write(f"(Calculated at λ = {st.session_state.monitoring_wavelength} nm and incidence = {st.session_state.incidence_angle_deg}°).")
         if st.button("Plot rs in Complex Plane"):
             if layer_multipliers_list is not None:
                 try:
                      with st.spinner("Generating complex plane plot..."):
                         st.session_state.fig_complex = plot_complex_rs(results, params_used_for_plots, layer_multipliers_list)
                         st.pyplot(st.session_state.fig_complex)
                 except Exception as e_complex:
                      st.error(f"Error plotting complex rs: {e_complex}")
                      st.error(traceback.format_exc())
             else:
                 st.warning("Please run a successful calculation first.")

         elif st.session_state.fig_complex:
               st.pyplot(st.session_state.fig_complex)
         else:
               st.info("Click the button above to generate the plot (requires successful calculation).")


    if st.session_state.export_excel:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export")
        if layer_multipliers_list is not None:
            try:
                with st.spinner("Preparing Excel file..."):
                     excel_data = generate_excel_output(results, params_used_for_plots, layer_multipliers_list)
                     num_layers_export = len(layer_multipliers_list) if layer_multipliers_list else 0
                     now = datetime.datetime.now()
                     timestamp = now.strftime("%Y%m%d-%H%M%S")
                     excel_filename = f"ThinFilm_Results_{num_layers_export}L_{timestamp}.xlsx"

                     st.sidebar.download_button(
                         label="📥 Download Results (Excel)",
                         data=excel_data,
                         file_name=excel_filename,
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         use_container_width=True
                     )
            except Exception as e_excel:
                st.sidebar.error(f"Error creating Excel file: {e_excel}")
                st.sidebar.error(traceback.format_exc())
        else:
             st.sidebar.warning("Run calculation first to enable export.")


elif run_calculation:
     st.warning("Calculation could not be completed. Please check parameters and error messages.")
else:
    st.info("Configure parameters in the sidebar and click 'Run Calculation'.")

st.sidebar.markdown("---")
st.sidebar.caption("Thin Film Calculator v1.7-en")
