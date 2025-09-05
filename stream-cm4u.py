import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
import io
import os
from scipy.interpolate import interp1d
import numba
import traceback
import logging

# ==============================================================================
# CONFIGURATION DU LOGGER
# ==============================================================================

def setup_logger():
    """Configure un logger pour capturer les événements dans un stream en mémoire."""
    if 'log_stream' not in st.session_state:
        st.session_state.log_stream = io.StringIO()
    
    logger = logging.getLogger('ThinFilmCalculator')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        stream_handler = logging.StreamHandler(st.session_state.log_stream)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

# ==============================================================================
# GESTION DES DONNÉES MATÉRIAUX
# ==============================================================================

@st.cache_data
def load_material_data(filepath="indices.xlsx", _logger=None):
    """
    Charge tous les matériaux depuis les onglets d'un fichier Excel.
    Note: requiert l'installation de 'openpyxl' (pip install openpyxl).
    """
    logger = _logger or logging.getLogger('ThinFilmCalculator')
    material_data = {}
    if not os.path.isfile(filepath):
        logger.warning(f"Fichier '{filepath}' non trouvé. Seuls les indices personnalisés seront disponibles.")
        return {}
        
    try:
        xls = pd.ExcelFile(filepath)
        logger.info(f"Lecture du fichier Excel '{filepath}', {len(xls.sheet_names)} onglets trouvés.")
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                df.columns = [str(col).strip() for col in df.columns]
                
                wl_col = next((col for col in df.columns if 'wavelength' in col.lower() or 'λ' in col.lower()), df.columns[0])
                n_col = next((col for col in df.columns if col.lower().startswith('n')), df.columns[1])
                k_col = next((col for col in df.columns if col.lower().startswith('k')), df.columns[2])

                df = df[[wl_col, n_col, k_col]].rename(columns={wl_col: 'wl', n_col: 'n', k_col: 'k'})
                df = df.sort_values(by='wl').dropna()

                if df.empty:
                    logger.warning(f"Aucune donnée valide dans l'onglet '{sheet_name}'. Onglet ignoré.")
                    continue

                material_name = sheet_name
                material_data[material_name] = {
                    'wl': df['wl'].values, 'n': df['n'].values, 'k': df['k'].values,
                    'interpolator_n': interp1d(df['wl'], df['n'], bounds_error=False, fill_value="extrapolate"),
                    'interpolator_k': interp1d(df['wl'], df['k'], bounds_error=False, fill_value="extrapolate")
                }
                logger.info(f"Matériau '{material_name}' chargé avec succès ({len(df)} points).")
            except Exception as e:
                logger.error(f"Erreur de lecture de l'onglet '{sheet_name}': {e}")
        return material_data
    except Exception as e:
        logger.error(f"Impossible de lire le fichier Excel '{filepath}'. Assurez-vous que 'openpyxl' est installé.", exc_info=True)
        return {}


def get_dispersive_index(material_name, wavelength_nm, material_database):
    """
    Calcule l'indice complexe interpolé.
    Convention: n_complexe = n - ik, où k (extinction) > 0 pour l'absorption.
    """
    if material_name not in material_database:
        raise ValueError(f"Matériau '{material_name}' non trouvé dans la base de données chargée.")
    
    data = material_database[material_name]
    n = data['interpolator_n'](wavelength_nm)
    k = data['interpolator_k'](wavelength_nm)
    return n - 1j * k

# --- Matériaux de substrat (Sellmeier) ---
SELLMEIER_COEFFS = {
    "N-BK7": {"B1": 1.03961212, "C1": 0.00600069867, "B2": 0.231792344, "C2": 0.0200179144, "B3": 1.01046945, "C3": 103.560653},
    "Silice fondue": {"B1": 0.6961663, "C1": 0.0684043**2, "B2": 0.4079426, "C2": 0.1162414**2, "B3": 0.8974794, "C3": 9.896161**2},
    "Saphir (o)": {"B1": 1.4313493, "C1": 0.0726631**2, "B2": 0.65054713, "C2": 0.1193242**2, "B3": 5.3414021, "C3": 18.028251**2},
}
SUBSTRATE_MAPPING = { "N-BK7": "N-BK7", "Silice": "Silice fondue", "Saphir": "Saphir (o)"}

def get_sellmeier_index(material_name, wavelength_nm):
    """Calcule l'indice de réfraction pour un substrat via la formule de Sellmeier."""
    coeffs = SELLMEIER_COEFFS[material_name]
    lambda_sq = (wavelength_nm / 1000.0)**2
    n_sq = 1 + sum( (coeffs[f"B{i+1}"] * lambda_sq) / (lambda_sq - coeffs[f"C{i+1}"]) for i in range(3))
    return np.sqrt(n_sq)


# ==============================================================================
# CŒUR DE CALCUL (OPTIMISÉ AVEC NUMBA)
# ==============================================================================

@numba.jit(nopython=True, cache=True)
def calculate_transfer_matrix(polarization, wl, angle_rad, n_layer_complex, thickness):
    alpha = np.sin(angle_rad)
    sqrt_term_sq = n_layer_complex**2 - alpha**2
    if np.real(sqrt_term_sq) < -1e-12: sqrt_val = 1j * np.sqrt(-sqrt_term_sq)
    elif abs(np.real(sqrt_term_sq)) < 1e-12 and abs(np.imag(sqrt_term_sq)) < 1e-12: sqrt_val = 1e-6 + 0j
    else: sqrt_val = np.sqrt(sqrt_term_sq)
    if polarization == 'p':
        eta = n_layer_complex**2 / sqrt_val if abs(sqrt_val) > 1e-9 else np.inf + 0j
    else: eta = sqrt_val
    phi = (2 * np.pi / wl) * sqrt_val * thickness
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    sin_phi_over_eta = (1j / eta) * sin_phi if abs(eta) > 1e-12 else (np.inf + 0j if abs(sin_phi) > 1e-9 else 0j)
    return np.array([[cos_phi, sin_phi_over_eta], [1j * eta * sin_phi, cos_phi]], dtype=np.complex128)

@numba.jit(nopython=True, cache=True)
def calculate_admittances(polarization, angle_rad, n_inc, n_sub):
    alpha = np.sin(angle_rad)
    eta_inc, eta_sub = 0j, 0j
    for medium_idx, n_medium in enumerate([np.complex128(n_inc), np.complex128(n_sub)]):
        sqrt_term_sq = n_medium**2 - alpha**2
        if np.real(sqrt_term_sq) < -1e-12: sqrt_val = 1j * np.sqrt(-sqrt_term_sq)
        elif abs(np.real(sqrt_term_sq)) < 1e-12 and abs(np.imag(sqrt_term_sq)) < 1e-12: sqrt_val = 1e-6 + 0j
        else: sqrt_val = np.sqrt(sqrt_term_sq)
        if polarization == 'p': eta = n_medium**2 / sqrt_val if abs(sqrt_val) > 1e-9 else np.inf + 0j
        else: eta = sqrt_val
        if medium_idx == 0: eta_inc = eta
        else: eta_sub = eta
    return eta_inc, eta_sub

# ==============================================================================
# FONCTIONS DE CALCUL PRINCIPALES
# ==============================================================================

def calculate_global_RT(wavelengths, angles_rad, nH_id, nL_id, nSub_id, p_thick, fin_sub, materials_db, incident_medium_index=1.0):
    RT = np.zeros((len(wavelengths), len(angles_rad), 4))
    is_dispersive_h = isinstance(nH_id, str)
    is_dispersive_l = isinstance(nL_id, str)
    is_dispersive_sub = isinstance(nSub_id, str)

    for i_wl, wl in enumerate(wavelengths):
        nH_at_wl = get_dispersive_index(nH_id, wl, materials_db) if is_dispersive_h else nH_id
        nL_at_wl = get_dispersive_index(nL_id, wl, materials_db) if is_dispersive_l else nL_id
        nSub_at_wl = get_sellmeier_index(nSub_id, wl) if is_dispersive_sub else nSub_id

        for i_ang, ang_rad in enumerate(angles_rad):
            M_s = np.eye(2, dtype=np.complex128)
            M_p = np.eye(2, dtype=np.complex128)
            for i_layer, thickness in enumerate(p_thick):
                n_layer = nH_at_wl if i_layer % 2 == 0 else nL_at_wl
                M_s = calculate_transfer_matrix('s', wl, ang_rad, n_layer, thickness) @ M_s
                M_p = calculate_transfer_matrix('p', wl, ang_rad, n_layer, thickness) @ M_p
            
            eta_inc_s, eta_sub_s = calculate_admittances('s', ang_rad, incident_medium_index, nSub_at_wl)
            eta_inc_p, eta_sub_p = calculate_admittances('p', ang_rad, incident_medium_index, nSub_at_wl)
            
            denom_s = (eta_inc_s * M_s[0, 0] + eta_sub_s * M_s[1, 1] + eta_inc_s * eta_sub_s * M_s[0, 1] + M_s[1, 0])
            rs_inf = (eta_inc_s * M_s[0, 0] - eta_sub_s * M_s[1, 1] + eta_inc_s * eta_sub_s * M_s[0, 1] - M_s[1, 0]) / denom_s if abs(denom_s) > 1e-12 else 1.0
            ts_inf = 2 * eta_inc_s / denom_s if abs(denom_s) > 1e-12 else 0.0
            Rs_inf = np.abs(rs_inf)**2
            Ts_inf = (np.real(eta_sub_s) / np.real(eta_inc_s)) * np.abs(ts_inf)**2 if abs(np.real(eta_inc_s)) > 1e-12 else 0.0

            denom_p = (eta_inc_p * M_p[0, 0] + eta_sub_p * M_p[1, 1] + eta_inc_p * eta_sub_p * M_p[0, 1] + M_p[1, 0])
            rp_inf = (eta_inc_p * M_p[0, 0] - eta_sub_p * M_p[1, 1] + eta_inc_p * eta_sub_p * M_p[0, 1] - M_p[1, 0]) / denom_p if abs(denom_p) > 1e-12 else 1.0
            tp_inf = 2 * eta_inc_p / denom_p if abs(denom_p) > 1e-12 else 0.0
            Rp_inf = np.abs(rp_inf)**2
            Tp_inf = (np.real(eta_sub_p) / np.real(eta_inc_p)) * np.abs(tp_inf)**2 if abs(np.real(eta_inc_p)) > 1e-12 else 0.0

            if fin_sub:
                eta_inc_rev_s, eta_sub_rev_s = calculate_admittances('s', ang_rad, nSub_at_wl, incident_medium_index)
                Rb_s = np.abs((eta_inc_rev_s - eta_sub_rev_s) / (eta_inc_rev_s + eta_sub_rev_s))**2 if abs(eta_inc_rev_s + eta_sub_rev_s) > 1e-12 else 1.0
                denom_s_corr = 1 - Rs_inf * Rb_s
                Rs = Rs_inf + (Ts_inf * Rb_s * Ts_inf / denom_s_corr) if abs(denom_s_corr) > 1e-12 else Rs_inf
                Ts = (Ts_inf * (1 - Rb_s) / denom_s_corr) if abs(denom_s_corr) > 1e-12 else Ts_inf * (1 - Rb_s)
                
                eta_inc_rev_p, eta_sub_rev_p = calculate_admittances('p', ang_rad, nSub_at_wl, incident_medium_index)
                Rb_p = np.abs((eta_inc_rev_p - eta_sub_rev_p) / (eta_inc_rev_p + eta_sub_rev_p))**2 if abs(eta_inc_rev_p + eta_sub_rev_p) > 1e-12 else 1.0
                denom_p_corr = 1 - Rp_inf * Rb_p
                Rp = Rp_inf + (Tp_inf * Rb_p * Tp_inf / denom_p_corr) if abs(denom_p_corr) > 1e-12 else Rp_inf
                Tp = (Tp_inf * (1 - Rb_p) / denom_p_corr) if abs(denom_p_corr) > 1e-12 else Tp_inf * (1 - Rb_p)
            else:
                Rs, Ts, Rp, Tp = Rs_inf, Ts_inf, Rp_inf, Tp_inf
            
            RT[i_wl, i_ang, :] = [np.clip(np.nan_to_num(v), 0, 1) for v in [Rs, Rp, Ts, Tp]]
    return RT

def calculate_stack_properties(nH_id, nL_id, nSub_id, l0, stack_string, wl_range, wl_step, ang_range, ang_step, angle_deg, mon_step, fin_sub, mon_wl, materials_db):
    multipliers = [float(e) for e in stack_string.split(',') if e.strip()]
    if not multipliers: raise ValueError("La définition de l'empilement est vide.")

    nH_for_thickness = get_dispersive_index(nH_id, l0, materials_db) if isinstance(nH_id, str) else nH_id
    nL_for_thickness = get_dispersive_index(nL_id, l0, materials_db) if isinstance(nL_id, str) else nL_id

    p_thick = []
    for i, m in enumerate(multipliers):
        n_ref = nH_for_thickness if i % 2 == 0 else nL_for_thickness
        if abs(np.real(n_ref)) < 1e-9: raise ValueError(f"La partie réelle de l'indice est nulle pour la couche {i+1}.")
        p_thick.append(m * l0 / (4 * np.real(n_ref)))

    wavelengths = np.arange(wl_range[0], wl_range[1] + wl_step, wl_step)
    angles_rad = np.radians(np.arange(ang_range[0], ang_range[1] + ang_step, ang_step))
    RT_spectral = calculate_global_RT(wavelengths, [np.radians(angle_deg)], nH_id, nL_id, nSub_id, p_thick, fin_sub, materials_db)
    RT_angular = calculate_global_RT([l0], angles_rad, nH_id, nL_id, nSub_id, p_thick, fin_sub, materials_db)

    # Calculs pour le monitoring
    nH_mon = get_dispersive_index(nH_id, mon_wl, materials_db) if isinstance(nH_id, str) else nH_id
    nL_mon = get_dispersive_index(nL_id, mon_wl, materials_db) if isinstance(nL_id, str) else nL_id
    nSub_mon = get_sellmeier_index(nSub_id, mon_wl) if isinstance(nSub_id, str) else nSub_id
    
    def get_transmission(matrix, n_sub, fin_sub_flag, angle_r, inc_med_idx=1.0):
        eta_i_s, eta_s_s = calculate_admittances('s', angle_r, inc_med_idx, n_sub)
        denom = (eta_i_s * matrix[0, 0] + eta_s_s * matrix[1, 1] + eta_i_s * eta_s_s * matrix[0, 1] + matrix[1, 0])
        ts_inf = 2 * eta_i_s / denom if abs(denom) > 1e-12 else 0.0
        Ts_inf = (np.real(eta_s_s) / np.real(eta_i_s)) * np.abs(ts_inf)**2 if abs(np.real(eta_i_s)) > 1e-12 else 0.0
        if not fin_sub_flag: return Ts_inf
        
        rs_inf = (eta_i_s * matrix[0, 0] - eta_s_s * matrix[1, 1] + eta_i_s * eta_s_s * matrix[0, 1] - matrix[1, 0]) / denom if abs(denom) > 1e-12 else 1.0
        Rs_inf = np.abs(rs_inf)**2
        eta_i_rev, eta_s_rev = calculate_admittances('s', angle_r, n_sub, inc_med_idx)
        Rb = np.abs((eta_i_rev - eta_s_rev) / (eta_i_rev + eta_s_rev))**2 if abs(eta_i_rev + eta_s_rev) > 1e-12 else 1.0
        denom_corr = 1 - Rs_inf * Rb
        return (Ts_inf * (1 - Rb) / denom_corr) if abs(denom_corr) > 1e-12 else Ts_inf * (1 - Rb)

    angle_rad_mon = np.radians(angle_deg)
    monitoring_t = [0.0]
    monitoring_T = [get_transmission(np.eye(2), nSub_mon, fin_sub, angle_rad_mon)]
    
    M_cumul = np.eye(2, dtype=np.complex128)
    c_thick = 0.0
    for i_layer, layer_thickness in enumerate(p_thick):
        n_layer = nH_mon if i_layer % 2 == 0 else nL_mon
        num_steps = max(1, int(np.ceil(layer_thickness / mon_step)))
        actual_step = layer_thickness / num_steps
        
        for _ in range(num_steps):
            M_slice = calculate_transfer_matrix('s', mon_wl, angle_rad_mon, n_layer, actual_step)
            M_cumul = M_slice @ M_cumul
            c_thick += actual_step
            monitoring_t.append(c_thick)
            monitoring_T.append(get_transmission(M_cumul, nSub_mon, fin_sub, angle_rad_mon))

    return {
        'wavelengths': wavelengths, 'angles_deg_angular': np.degrees(angles_rad),
        'Rs_spectral': RT_spectral[:, 0, 0], 'Rp_spectral': RT_spectral[:, 0, 1],
        'Ts_spectral': RT_spectral[:, 0, 2], 'Tp_spectral': RT_spectral[:, 0, 3],
        'Rs_angular': RT_angular[0, :, 0], 'Rp_angular': RT_angular[0, :, 1],
        'Ts_angular': RT_angular[0, :, 2], 'Tp_angular': RT_angular[0, :, 3],
        'monitoring_thickness': np.array(monitoring_t),
        'monitoring_transmittance': np.clip(np.nan_to_num(monitoring_T), 0, 1),
        'physical_thicknesses': p_thick,
        'nSub_for_monitoring': nSub_mon, 'nH_for_monitoring': nH_mon, 'nL_for_monitoring': nL_mon,
        'nH_for_thickness': nH_for_thickness, 'nL_for_thickness': nL_for_thickness,
    }, multipliers

# ==============================================================================
# Fonctions de traçage et d'export
# ==============================================================================

def plot_spectral_results(res, params):
    """Génère le graphique spectral interactif avec Plotly."""
    fig = go.Figure()
    wl = res['wavelengths']
    As = 1 - res['Rs_spectral'] - res['Ts_spectral']
    Ap = 1 - res['Rp_spectral'] - res['Tp_spectral']
    
    fig.add_trace(go.Scatter(x=wl, y=res['Rs_spectral'], name='Rs', mode='lines', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=wl, y=res['Rp_spectral'], name='Rp', mode='lines', line=dict(color='#1f77b4', dash='dash')))
    fig.add_trace(go.Scatter(x=wl, y=res['Ts_spectral'], name='Ts', mode='lines', line=dict(color='#d62728')))
    fig.add_trace(go.Scatter(x=wl, y=res['Tp_spectral'], name='Tp', mode='lines', line=dict(color='#d62728', dash='dash')))
    fig.add_trace(go.Scatter(x=wl, y=As, name='As (Abs)', mode='lines', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=wl, y=Ap, name='Ap (Abs)', mode='lines', line=dict(color='black', dash='dot')))

    fig.update_layout(
        title=f"Scan Spectral (Incidence à {params['angle_deg']:.1f}°)",
        xaxis_title="Longueur d'onde (nm)",
        yaxis_title='Réflectance / Transmittance / Absorptance',
        yaxis_range=[-0.05, 1.05],
        template='plotly_white',
        legend_title="Polarisation"
    )
    return fig

def plot_angular_results(res, params):
    """Génère le graphique angulaire interactif avec Plotly."""
    fig = go.Figure()
    angles = res['angles_deg_angular']
    As = 1 - res['Rs_angular'] - res['Ts_angular']
    Ap = 1 - res['Rp_angular'] - res['Tp_angular']
    
    fig.add_trace(go.Scatter(x=angles, y=res['Rs_angular'], name='Rs', mode='lines', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=angles, y=res['Rp_angular'], name='Rp', mode='lines', line=dict(color='#1f77b4', dash='dash')))
    fig.add_trace(go.Scatter(x=angles, y=res['Ts_angular'], name='Ts', mode='lines', line=dict(color='#d62728')))
    fig.add_trace(go.Scatter(x=angles, y=res['Tp_angular'], name='Tp', mode='lines', line=dict(color='#d62728', dash='dash')))
    fig.add_trace(go.Scatter(x=angles, y=As, name='As (Abs)', mode='lines', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=angles, y=Ap, name='Ap (Abs)', mode='lines', line=dict(color='black', dash='dot')))

    fig.update_layout(
        title=f"Scan Angulaire (λ = {params['l0']:.0f} nm)",
        xaxis_title="Angle d'incidence (degrés)",
        yaxis_title='Réflectance / Transmittance / Absorptance',
        yaxis_range=[-0.05, 1.05],
        template='plotly_white',
        legend_title="Polarisation"
    )
    return fig

def plot_index_and_monitoring(res, params, multipliers):
    """Génère le graphique du profil d'indice et du monitoring avec Plotly."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Profil d'indice
    nH_r, nL_r, nSub_r = np.real(res['nH_for_monitoring']), np.real(res['nL_for_monitoring']), np.real(res['nSub_for_monitoring'])
    x_coords, y_coords = [-50, 0], [nSub_r, nSub_r]
    current_t = 0.0
    if multipliers:
        real_indices = [nH_r if i % 2 == 0 else nL_r for i in range(len(multipliers))]
        for i, thickness in enumerate(res['physical_thicknesses']):
            x_coords.extend([current_t, current_t + thickness, None]) # 'None' to create gap
            y_coords.extend([real_indices[i], real_indices[i], None])
            current_t += thickness
    last_t = sum(res['physical_thicknesses'])
    x_coords.extend([last_t, last_t + 50]); y_coords.extend([1.0, 1.0])
    
    fig.add_trace(
        go.Scatter(x=x_coords, y=y_coords, name=f'n (réel) @ {params["mon_wl"]:.0f} nm', line=dict(color='green', width=3)),
        secondary_y=False,
    )

    # Courbe de monitoring
    t_sorted, T_sorted = res['monitoring_thickness'], res['monitoring_transmittance']
    fig.add_trace(
        go.Scatter(x=t_sorted, y=T_sorted, name='T (simulée)', line=dict(color='red', width=2)),
        secondary_y=True,
    )

    # Mise en forme
    fig.update_layout(
        title_text="Profil d'Indice & Courbe de Monitoring",
        template='plotly_white',
        xaxis_title='Épaisseur Cumulée (nm)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Indice de Réfraction (n)</b>", color='green', secondary_y=False)
    fig.update_yaxes(title_text="<b>Transmittance Monitoring</b>", color='red', secondary_y=True, range=[-0.05, 1.05])
    
    return fig

def plot_stack_structure(res, params, multipliers):
    """Génère le graphique de la structure de l'empilement avec Plotly."""
    nH_l0, nL_l0 = res['nH_for_thickness'], res['nL_for_thickness']
    p_thick = res['physical_thicknesses']
    num_layers = len(multipliers)
    layer_labels = [f"Couche {i+1}" for i in range(num_layers)]
    indices = [(nH_l0 if i % 2 == 0 else nL_l0) for i in range(num_layers)]
    colors = ['#aec7e8' if i % 2 == 0 else '#ff9896' for i in range(num_layers)]
    
    text_labels = [f"{thick:.1f} nm<br>n={np.real(idx):.3f} - i*{abs(np.imag(idx)):.4f}" for thick, idx in zip(p_thick, indices)]

    fig = go.Figure(go.Bar(
        x=p_thick,
        y=layer_labels,
        orientation='h',
        marker_color=colors,
        text=text_labels,
        textposition='inside',
        insidetextanchor='middle'
    ))

    fig.update_layout(
        title=f'Structure de l\'empilement (Indices @ {params["l0"]:.0f} nm)',
        xaxis_title='Épaisseur (nm)',
        yaxis_title='Couche',
        template='plotly_white',
        yaxis={'categoryorder':'array', 'categoryarray': layer_labels} # Correction de l'ordre d'affichage
    )
    return fig

def calculate_complex_rs_trajectory(res, params):
    """Calcule la trajectoire du coefficient de réflexion rs dans le plan complexe."""
    mon_wl, angle_rad_mon = params['mon_wl'], np.radians(params['angle_deg'])
    nH_mon, nL_mon, nSub_mon = res['nH_for_monitoring'], res['nL_for_monitoring'], res['nSub_for_monitoring']
    p_thick, mon_step = res['physical_thicknesses'], params['mon_step']
    
    eta_inc_s, eta_sub_s = calculate_admittances('s', angle_rad_mon, 1.0, nSub_mon)
    
    def get_rs(matrix):
        denom = (eta_inc_s * matrix[0, 0] + eta_sub_s * matrix[1, 1] + eta_inc_s * eta_sub_s * matrix[0, 1] + matrix[1, 0])
        return (eta_inc_s * matrix[0, 0] - eta_sub_s * matrix[1, 1] + eta_inc_s * eta_sub_s * matrix[0, 1] - matrix[1, 0]) / denom if abs(denom) > 1e-12 else 1.0

    rs_values = [get_rs(np.eye(2))]
    M_cumul = np.eye(2, dtype=np.complex128)
    for i_layer, thickness in enumerate(p_thick):
        n_layer = nH_mon if i_layer % 2 == 0 else nL_mon
        num_steps = max(1, int(np.ceil(thickness / mon_step)))
        actual_step = thickness / num_steps
        for _ in range(num_steps):
            M_slice = calculate_transfer_matrix('s', mon_wl, angle_rad_mon, n_layer, actual_step)
            M_cumul = M_slice @ M_cumul
            rs_values.append(get_rs(M_cumul))
            
    return np.array(rs_values)

def plot_complex_rs(res, params, multipliers):
    """Génère le graphique du plan complexe avec Plotly."""
    rs_values = calculate_complex_rs_trajectory(res, params)
    
    fig = go.Figure()
    # Trajectoire
    fig.add_trace(go.Scatter(
        x=np.real(rs_values), 
        y=np.imag(rs_values),
        mode='lines',
        line=dict(color='black', width=1),
        name='Trajectoire'
    ))
    # Point de départ
    fig.add_trace(go.Scatter(
        x=[np.real(rs_values[0])], 
        y=[np.imag(rs_values[0])],
        mode='markers',
        marker=dict(color='green', size=10, symbol='circle'),
        name='Substrat'
    ))
    # Point final
    fig.add_trace(go.Scatter(
        x=[np.real(rs_values[-1])], 
        y=[np.imag(rs_values[-1])],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Final'
    ))
    
    fig.update_layout(
        title=f'Plan Complexe rs (λ={params["mon_wl"]:.0f} nm, θ={params["angle_deg"]:.1f}°)',
        xaxis_title='Re(rs)',
        yaxis_title='Im(rs)',
        template='plotly_white',
        showlegend=True,
        xaxis=dict(scaleanchor="y", scaleratio=1), # Aspect ratio 1:1
    )
    return fig

def generate_excel_output(res, params, multipliers):
    """Génère un fichier Excel contenant tous les résultats de la simulation."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Onglet 1: Paramètres
        params_to_export = params.copy()
        params_to_export.pop('materials_db', None) # Retirer l'objet non sérialisable
        params_df = pd.DataFrame.from_dict(params_to_export, orient='index', columns=['Value'])
        params_df.to_excel(writer, sheet_name='Parameters')

        # Onglet 2: Données Spectrales
        spectral_df = pd.DataFrame({
            'Wavelength (nm)': res['wavelengths'],
            'Rs': res['Rs_spectral'], 'Rp': res['Rp_spectral'],
            'Ts': res['Ts_spectral'], 'Tp': res['Tp_spectral'],
            'As': 1 - res['Rs_spectral'] - res['Ts_spectral'],
            'Ap': 1 - res['Rp_spectral'] - res['Tp_spectral'],
        })
        spectral_df.to_excel(writer, sheet_name='Spectral Data', index=False)
        
        # Onglet 3: Données Angulaires
        angular_df = pd.DataFrame({
            'Angle (degrés)': res['angles_deg_angular'],
            'Rs': res['Rs_angular'], 'Rp': res['Rp_angular'],
            'Ts': res['Ts_angular'], 'Tp': res['Tp_angular'],
            'As': 1 - res['Rs_angular'] - res['Ts_angular'],
            'Ap': 1 - res['Rp_angular'] - res['Tp_angular'],
        })
        angular_df.to_excel(writer, sheet_name='Angular Data', index=False)
        
        # Onglet 4: Données de Monitoring
        monitoring_df = pd.DataFrame({
            'Épaisseur Cumulée (nm)': res['monitoring_thickness'],
            'Transmittance (T)': res['monitoring_transmittance']
        })
        monitoring_df.to_excel(writer, sheet_name='Monitoring Data', index=False)

        # Onglet 5: Détails des Couches
        layer_df = pd.DataFrame({
            'Couche #': range(1, len(multipliers) + 1),
            'Type': ['H' if i % 2 == 0 else 'L' for i in range(len(multipliers))],
            f'Indice @ {params["l0"]}nm': [f"{v.real:.4f} - i*{abs(v.imag):.4f}" for v in [res['nH_for_thickness'] if i % 2 == 0 else res['nL_for_thickness'] for i in range(len(multipliers))]],
            'Épaisseur (nm)': res['physical_thicknesses']
        })
        layer_df.to_excel(writer, sheet_name='Layer Details', index=False)
        
        # Onglet 6: Données du Plan Complexe (rs)
        rs_values = calculate_complex_rs_trajectory(res, params)
        complex_df = pd.DataFrame({
            'Re(rs)': np.real(rs_values),
            'Im(rs)': np.imag(rs_values)
        })
        complex_df.to_excel(writer, sheet_name='Complex Plane (rs)', index=False)

        # Ajustement automatique de la largeur des colonnes pour tous les onglets
        all_sheets = [
            ('Parameters', params_df), ('Spectral Data', spectral_df), 
            ('Angular Data', angular_df), ('Monitoring Data', monitoring_df), 
            ('Layer Details', layer_df), ('Complex Plane (rs)', complex_df)
        ]

        for sheet_name, df_to_check in all_sheets:
            worksheet = writer.sheets[sheet_name]
            if sheet_name == 'Parameters':
                # Gérer la colonne d'index
                max_len = max((df_to_check.index.astype(str).map(len).max()), len(df_to_check.index.name or "")) + 2
                worksheet.set_column(0, 0, max_len)
                start_col = 1
            else:
                start_col = 0
            
            # Gérer les colonnes de données
            for i, col in enumerate(df_to_check.columns, start=start_col):
                # Inclure la longueur de l'en-tête et la longueur maximale des données
                max_len = max(len(str(col)), df_to_check[col].astype(str).map(len).max()) + 2
                worksheet.set_column(i, i, max_len)

    output.seek(0)
    return output


# ==============================================================================
# INTERFACE UTILISATEUR (UI)
# ==============================================================================

def initialize_session_state():
    defaults = {
        'h_type': 'Personnalisé', 'l_type': 'Personnalisé',
        'nH_r': 2.3, 'nH_i': 0.0, 'nL_r': 1.45, 'nL_i': 0.0,
        'substrate_choice': "Personnalisé", 'nSub_custom': 1.45, 'l0': 550.0,
        'stack_string': "1,1,1,1,1,1,1,1", 'incidence_angle_deg': 0.0,
        'finite_substrate': False, 'wl_range_start': 400.0, 'wl_range_end': 700.0,
        'wl_step': 1.0, 'ang_range_start': 0.0, 'ang_range_end': 89.0, 'ang_step': 1.0,
        'monitoring_wavelength': 550.0, 'monitoring_step': 0.2, 'export_excel': False,
        'results': None, 'h_material_file': None, 'l_material_file': None,
        'calc_params': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_sidebar(material_list):
    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar.expander("Matériaux des Couches", expanded=True):
        st.subheader("Matériau Haut Indice (H)")
        st.radio("Type H", ["Personnalisé", "Dispersif (Fichier)"], key="h_type", horizontal=True)
        if st.session_state.h_type == "Personnalisé":
            c1, c2 = st.columns(2)
            c1.number_input("n (réel)", key="nH_r", step=0.01, format="%.4f")
            c2.number_input("k (imag)", key="nH_i", step=0.0001, format="%.4f", min_value=0.0)
        else:
            st.selectbox("Fichier Matériau H", options=material_list, key="h_material_file")
        
        st.subheader("Matériau Bas Indice (L)")
        st.radio("Type L", ["Personnalisé", "Dispersif (Fichier)"], key="l_type", horizontal=True)
        if st.session_state.l_type == "Personnalisé":
            c1, c2 = st.columns(2)
            c1.number_input("n (réel)", key="nL_r", step=0.01, format="%.4f")
            c2.number_input("k (imag)", key="nL_i", step=0.0001, format="%.4f", min_value=0.0)
        else:
            st.selectbox("Fichier Matériau L", options=material_list, key="l_material_file")

    with st.sidebar.expander("Substrat et Géométrie", expanded=True):
        st.selectbox("Matériau Substrat", ["Personnalisé"] + list(SUBSTRATE_MAPPING.keys()), key="substrate_choice")
        if st.session_state.substrate_choice == "Personnalisé":
            st.number_input("Indice Substrat (personnalisé)", key="nSub_custom", step=0.01, format="%.4f")
        st.number_input("λ centrale pour QWOT (nm)", key="l0", step=1.0, min_value=0.1)
        st.text_input("Définition (QWOT, ex: 1,1,2,1)", key="stack_string")
        st.slider("Angle d'incidence (degrés)", 0.0, 89.9, key="incidence_angle_deg", step=0.1)
        st.checkbox("Substrat Fini (réflexions incohérentes)", key="finite_substrate")
    
    with st.sidebar.expander("Plages de Calcul", expanded=False):
        st.subheader("Spectral")
        c1, c2, c3 = st.columns(3)
        c1.number_input("λ Début (nm)", key="wl_range_start")
        c2.number_input("λ Fin (nm)", key="wl_range_end")
        c3.number_input("Pas λ (nm)", key="wl_step", min_value=0.01)
        st.subheader("Angulaire")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Angle Début (°)", key="ang_range_start")
        c2.number_input("Angle Fin (°)", key="ang_range_end")
        c3.number_input("Pas Angle (°)", key="ang_step", min_value=0.01)

    with st.sidebar.expander("Monitoring Optique", expanded=False):
        st.number_input("λ de Monitoring (nm)", key="monitoring_wavelength")
        st.number_input("Pas du monitoring (nm)", key="monitoring_step", min_value=0.01, value=0.2, step=0.1, format="%.2f")

    st.sidebar.markdown("---")
    st.sidebar.header("Actions")
    run_button = st.sidebar.button("🚀 Lancer le calcul", use_container_width=True, type="primary")
    st.sidebar.checkbox("Activer l'export Excel", key="export_excel")
    return run_button

def display_results(params, multipliers):
    """Affiche les onglets avec les résultats."""
    st.subheader("Résultats de la Simulation")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Spectral & Angulaire", "🔬 Profil & Monitoring", "🏗️ Structure", "🌀 Plan Complexe (rs)", "📝 Log"])
    with tab1:
        st.plotly_chart(st.session_state.fig_spectral, use_container_width=True)
        st.plotly_chart(st.session_state.fig_angular, use_container_width=True)
    with tab2:
        st.plotly_chart(st.session_state.fig_profile_monitoring, use_container_width=True)
    with tab3:
        st.plotly_chart(st.session_state.fig_stack, use_container_width=True)
    with tab4:
        st.info(f"Trace le coefficient de réflexion `rs` (Pol-S) lors du dépôt. (λ={params['mon_wl']:.0f} nm, θ={params['angle_deg']:.1f}°)")
        if st.button("Générer le tracé du plan complexe"):
            with st.spinner("Génération du graphique..."):
                fig_complex = plot_complex_rs(st.session_state.results, params, multipliers)
                st.session_state.fig_complex = fig_complex
        
        if 'fig_complex' in st.session_state and st.session_state.fig_complex:
            st.plotly_chart(st.session_state.fig_complex, use_container_width=True)
    with tab5:
        st.info("Journal des événements de la session en cours.")
        log_content = st.session_state.log_stream.getvalue()
        st.code(log_content, language='log')
        st.download_button(
            label="Exporter le log",
            data=log_content,
            file_name=f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def main():
    st.set_page_config(page_title="Calculateur de Couches Minces", layout="wide")
    
    logger = setup_logger()
    
    if 'app_initialized' not in st.session_state:
        logger.info("="*50)
        logger.info("Initialisation de l'application.")
        initialize_session_state()
        st.session_state.app_initialized = True

    materials_database = load_material_data("indices.xlsx", _logger=logger)
    material_names = list(materials_database.keys()) if materials_database else []

    run_button = setup_sidebar(material_names)
    
    st.title("Calculateur de Couches Minces Avancé")

    if run_button:
        st.session_state.log_stream.truncate(0); st.session_state.log_stream.seek(0)
        logger.info("Bouton 'Lancer le calcul' cliqué.")
        
        # Effacer les anciens graphiques pour forcer la regénération
        for key in ['fig_spectral', 'fig_angular', 'fig_profile_monitoring', 'fig_stack', 'fig_complex']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.results = None # Clear previous results
        nH_id = (st.session_state.nH_r - 1j * st.session_state.nH_i) if st.session_state.h_type == 'Personnalisé' else st.session_state.h_material_file
        nL_id = (st.session_state.nL_r - 1j * st.session_state.nL_i) if st.session_state.l_type == 'Personnalisé' else st.session_state.l_material_file
        nSub_id = st.session_state.nSub_custom if st.session_state.substrate_choice == 'Personnalisé' else SUBSTRATE_MAPPING[st.session_state.substrate_choice]

        if (st.session_state.h_type != 'Personnalisé' and not nH_id) or \
           (st.session_state.l_type != 'Personnalisé' and not nL_id):
            st.error("Veuillez sélectionner un fichier de matériau ou choisir le mode 'Personnalisé'.")
            logger.error("Erreur de validation: Fichier de matériau non sélectionné.")
        else:
            with st.spinner("Calcul en cours..."):
                try:
                    logger.info("Début du calcul des propriétés de l'empilement.")
                    calc_params = {
                        'nH_id': nH_id, 'nL_id': nL_id, 'nSub_id': nSub_id,
                        'l0': st.session_state.l0, 'stack_string': st.session_state.stack_string,
                        'wl_range': (st.session_state.wl_range_start, st.session_state.wl_range_end),
                        'wl_step': st.session_state.wl_step,
                        'ang_range': (st.session_state.ang_range_start, st.session_state.ang_range_end),
                        'ang_step': st.session_state.ang_step,
                        'angle_deg': st.session_state.incidence_angle_deg,
                        'mon_step': st.session_state.monitoring_step,
                        'fin_sub': st.session_state.finite_substrate,
                        'mon_wl': st.session_state.monitoring_wavelength,
                        'materials_db': materials_database
                    }
                    st.session_state.calc_params = calc_params
                    
                    results, multipliers = calculate_stack_properties(**calc_params)
                    
                    st.session_state.results = results
                    st.session_state.layer_multipliers_list = multipliers
                    logger.info("Calcul terminé avec succès.")

                    logger.info("Génération des graphiques...")
                    params_for_plots = {k: calc_params[k] for k in ['angle_deg', 'l0', 'mon_wl', 'mon_step']}
                    st.session_state.fig_spectral = plot_spectral_results(results, params_for_plots)
                    st.session_state.fig_angular = plot_angular_results(results, params_for_plots)
                    st.session_state.fig_profile_monitoring = plot_index_and_monitoring(results, params_for_plots, multipliers)
                    st.session_state.fig_stack = plot_stack_structure(results, params_for_plots, multipliers)
                    logger.info("Graphiques générés.")
                    
                    st.success("Calcul terminé!")

                except Exception as e:
                    logger.error(f"Une erreur est survenue durant le calcul.", exc_info=True)
                    st.error(f"Une erreur est survenue : {e}")
    
    if st.session_state.results:
        display_results(st.session_state.calc_params, st.session_state.layer_multipliers_list)
    else:
        st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer le calcul'.")

    if st.session_state.export_excel and st.session_state.results:
        logger.info("Préparation de l'export Excel.")
        excel_data = generate_excel_output(st.session_state.results, st.session_state.calc_params, st.session_state.layer_multipliers_list)
        st.sidebar.download_button(
            label="📥 Télécharger Fichier Excel",
            data=excel_data,
            file_name=f"Resultats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

if __name__ == "__main__":
    main()

