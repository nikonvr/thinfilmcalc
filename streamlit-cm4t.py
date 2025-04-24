import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or another suitable backend, e.g., 'Qt5Agg'
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import mplcursors
from scipy.interpolate import make_interp_spline
import pandas as pd
import os
import datetime
from matplotlib.animation import FuncAnimation
import time

def calcul_empilement(nH, nL, nSub, l0, emp_str, l_range, l_step, a_range, a_step, inc, n_inter, substrat_fini, lambda_monitoring):
    """Calcule la réflectance et la transmittance d'un empilement de couches minces."""
    l, theta_inc_spectral = np.arange(l_range[0], l_range[1] + l_step, l_step), np.radians(inc)
    l_ang, theta_inc_ang = np.array([l0]), np.radians(np.arange(a_range[0], a_range[1] + a_step, a_step))
    emp = [float(e) for e in emp_str.split(',')]
    ep = [e * l0 / (4 * np.real(nH if i % 2 == 0 else nL)) for i, e in enumerate(emp)]

    # Dictionnaire pour stocker les matrices de transfert à chaque interface
    matrices_stockees = {}

    def calcul_RT(longueurs_onde, angles):
        """Calcule R et T pour différentes longueurs d'onde et angles."""
        RT = np.zeros((len(longueurs_onde), len(angles), 4))

        for i_l, l in enumerate(longueurs_onde):
            for i_a, theta in enumerate(angles):
                alpha = np.sin(theta)

                def calcul_M(pol):
                    """Calcule la matrice de transfert M pour une polarisation donnée."""
                    M = np.eye(2, dtype=complex)

                    for i, e in enumerate(emp):
                        Ni = nH if i % 2 == 0 else nL
                        eta = Ni**2 / np.sqrt(Ni**2 - alpha**2) if pol == 'p' else np.sqrt(Ni**2 - alpha**2)
                        phi = (2 * np.pi / l) * (np.sqrt(Ni**2 - alpha**2) * ep[i])

                        key = (i, l, alpha, pol)
                        if key not in matrices_stockees:
                            matrices_stockees[key] = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                                                [1j * eta * np.sin(phi), np.cos(phi)]])
                        M = matrices_stockees[key] @ M

                    etainc = 1 / np.sqrt(1 - alpha**2) if pol == 'p' else np.sqrt(1 - alpha**2)
                    etasub = nSub**2 / np.sqrt(nSub**2 - alpha**2) if pol == 'p' else np.sqrt(nSub**2 - alpha**2)
                    return M, etainc, etasub

                # Calcul pour la polarisation s et p
                Ms, etainc_s, etasub_s = calcul_M('s')
                Mp, etainc_p, etasub_p = calcul_M('p')

                rs_infini = (etainc_s * Ms[0, 0] - etasub_s * Ms[1, 1] + etainc_s * etasub_s * Ms[0, 1] - Ms[1, 0]) / (etainc_s * Ms[0, 0] + etasub_s * Ms[1, 1] + etainc_s * etasub_s * Ms[0, 1] + Ms[1, 0])
                ts_infini = 2 * etainc_s / (etainc_s * Ms[0, 0] + etasub_s * Ms[1, 1] + etainc_s * etasub_s * Ms[0, 1] + Ms[1, 0])
                Rs_infini = np.abs(rs_infini)**2
                Ts_infini = (np.real(etasub_s) / np.real(etainc_s)) * np.abs(ts_infini)**2

                rp_infini = (etainc_p * Mp[0, 0] - etasub_p * Mp[1, 1] + etainc_p * etasub_p * Mp[0, 1] - Mp[1, 0]) / (etainc_p * Mp[0, 0] + etasub_p * Mp[1, 1] + etainc_p * etasub_p * Mp[0, 1] + Mp[1, 0])
                tp_infini = 2 * etainc_p / (etainc_p * Mp[0, 0] + etasub_p * Mp[1, 1] + etainc_p * etasub_p * Mp[0, 1] + Mp[1, 0])
                Rp_infini = np.abs(rp_infini)**2
                Tp_infini = (np.real(etasub_p) / np.real(etainc_p)) * np.abs(tp_infini)**2

                if substrat_fini:
                    Rb_s = np.abs((etainc_s - etasub_s) / (etainc_s + etasub_s))**2
                    Rb_p = np.abs((etainc_p - etasub_p) / (etainc_p + etasub_p))**2
                    Rs = (Rs_infini + Rb_s * Ts_infini * Tp_infini / (1 - Rs_infini * Rb_s))
                    Ts = ((1 - Rb_s) * Ts_infini / (1 - Rs_infini * Rb_s))
                    Rp = (Rp_infini + Rb_p * Tp_infini * Ts_infini / (1 - Rp_infini * Rb_p))
                    Tp = ((1 - Rb_p) * Tp_infini / (1 - Rp_infini * Rb_p))
                else:
                    Rs, Ts, Rp, Tp = Rs_infini, Ts_infini, Rp_infini, Tp_infini

                RT[i_l, i_a, 0] = Rs
                RT[i_l, i_a, 2] = Ts
                RT[i_l, i_a, 1] = Rp
                RT[i_l, i_a, 3] = Tp

        return RT

    RT_s = calcul_RT(l, [theta_inc_spectral])
    RT_a = calcul_RT(l_ang, theta_inc_ang)

    def calcul_T_couche(longueur_onde, angle, substrat_fini):
        """Calcule la transmittance à travers chaque couche."""
        alpha = np.sin(angle)
        matrices_cumul = {}

        # OPTIMISATION 4 : Définition de calcul_M_T *en dehors* de la boucle (même si elle est simple).
        def calcul_M_T(pol):
            """Calcule la matrice de transfert pour le calcul de la transmittance (version simplifiée)."""
            #Précalcul simplifié (on a besoin de alpha, mais pas de etainc, etasub ici)
            M = np.eye(2, dtype=complex)
            return M
        
        # OPTIMISATION 4 (suite): Initialisation simplifiée
        transmissions = [1.0]
        transmissions_inter = []


        # Calcul des matrices cumulées jusqu'à la *fin* de chaque couche.
        for i in range(len(emp)):
            M_cumul = calcul_M_T('s')  # Polarisation 's' est suffisante
            for j in range(i + 1):
                Ni = nH if j % 2 == 0 else nL
                phi = (2 * np.pi / longueur_onde) * (np.sqrt(Ni**2 - alpha**2) * ep[j])
                eta = Ni # OPTIMISATION 4 (suite)
                M_temp = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                    [1j * eta * np.sin(phi), np.cos(phi)]])
                M_cumul = M_temp @ M_cumul
            matrices_cumul[i] = M_cumul


        # OPTIMISATION 1: Suppression de la boucle externe redondante dans calcul_T_couche.
        for i in range(len(emp)):
            #Calcul de la transmittance aux points intermediaires
            for k in range(1, n_inter + 1):
                # OPTIMISATION 1 (suite): Utilisation directe de matrices_cumul.
                M_avant = matrices_cumul[i-1] if i > 0 else np.eye(2, dtype=complex)

                Ni = nH if i % 2 == 0 else nL
                ep_temp = ep[i] * k / (n_inter + 1)  # Epaisseur partielle
                phi = (2 * np.pi / longueur_onde) * (np.sqrt(Ni**2 - alpha**2) * ep_temp)
                eta = Ni # OPTIMISATION 4 (suite)
                M_segment = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                      [1j * eta * np.sin(phi), np.cos(phi)]])
                M = M_segment @ M_avant #on applique la matrice de l'épaisseur partielle

                #Calcul de T avec simplification
                etainc = 1. / np.sqrt(1. - alpha**2) if  np.isclose(angle, np.pi/2.) else np.sqrt(1. - alpha**2)  # Gère incidence normale/quasi-normale
                etasub = nSub**2 / np.sqrt(nSub**2 - alpha**2) if  np.isclose(angle, np.pi/2.) else np.sqrt(nSub**2 - alpha**2)

                ts_infini = 2 * etainc / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0])
                Ts_infini = (np.real(etasub) / np.real(etainc)) * np.abs(ts_infini) ** 2

                if substrat_fini:
                    Rb = np.abs((etainc - etasub) / (etainc + etasub)) ** 2
                    # Formule simplifiée
                    T = ((1 - Rb) * Ts_infini) / (1 - np.abs((etainc * M[0, 0] - etasub * M[1, 1] + etainc * etasub * M[0, 1] - M[1, 0]) / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0])) ** 2 * Rb)
                else :
                    T = Ts_infini

                transmissions_inter.append(T)

            #Calcul de la transmittance en fin de couche
            M = matrices_cumul[i]
            etainc = 1. / np.sqrt(1. - alpha**2) if  np.isclose(angle, np.pi/2.) else np.sqrt(1. - alpha**2)
            etasub = nSub**2 / np.sqrt(nSub**2 - alpha**2) if  np.isclose(angle, np.pi/2.) else np.sqrt(nSub**2 - alpha**2)

            ts_infini = 2 * etainc / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0])
            Ts_infini = (np.real(etasub) / np.real(etainc)) * np.abs(ts_infini) ** 2
            if substrat_fini:
                Rb = np.abs((etainc - etasub) / (etainc + etasub))**2
                T = ((1-Rb)*Ts_infini)/(1 - np.abs((etainc * M[0, 0] - etasub * M[1, 1] + etainc * etasub * M[0, 1] - M[1, 0])/(etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0]))**2 * Rb)
            else:
                T = Ts_infini

            transmissions.append(T)


        return transmissions, transmissions_inter

    transmissions, transmissions_inter = calcul_T_couche(lambda_monitoring, theta_inc_spectral, substrat_fini)

    return {'l': l, 'inc': np.array([inc]), 'Rs_s': RT_s[:, :, 0], 'Rp_s': RT_s[:, :, 1],
            'Ts_s': RT_s[:, :, 2], 'Tp_s': RT_s[:, :, 3], 'l_a': l_ang, 'inc_a': np.degrees(theta_inc_ang),
            'Rs_a': RT_a[:, :, 0], 'Rp_a': RT_a[:, :, 1], 'Ts_a': RT_a[:, :, 2], 'Tp_a': RT_a[:, :, 3],
            'transmissions': transmissions, 'transmissions_inter': transmissions_inter}, ep

def tracer_graphiques(res, ep, nH_r, nH_i, nL_r, nL_i, nSub, emp_str, inc, n_inter, lambda_monitoring):
    """Trace les graphiques de R, T et l'empilement."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.canvas.manager.set_window_title('calcul_CM_Fab')
    emp = [float(e) for e in emp_str.split(',')]
    ep_cum = np.cumsum(ep)

    traces = [
        {'x': res['l'], 'y': res['Rs_s'][:, 0], 'label': 'Rs', 'linestyle': '-', 'xlabel': 'Longueur d\'onde (nm)', 'ylabel': 'Reflectance / Transmittance', 'title': f'Tracé spectral (incidence {inc}°) ', 'cursor_format': "λ={x:.2f} nm\n{label}={y:.3f}"},
        {'x': res['l'], 'y': res['Rp_s'][:, 0], 'label': 'Rp', 'linestyle': '--', 'xlabel': '', 'ylabel': '', 'title': '', 'cursor_format': "λ={x:.2f} nm\n{label}={y:.3f}"},
        {'x': res['l'], 'y': res['Ts_s'][:, 0], 'label': 'Ts', 'linestyle': '-', 'xlabel': '', 'ylabel': '', 'title': '', 'cursor_format': "λ={x:.2f} nm\n{label}={y:.3f}"},
        {'x': res['l'], 'y': res['Tp_s'][:, 0], 'label': 'Tp', 'linestyle': '--', 'xlabel': '', 'ylabel': '', 'title': '', 'cursor_format': "λ={x:.2f} nm\n{label}={y:.3f}"},
        {'x': res['inc_a'], 'y': res['Rs_a'][0, :], 'label': 'Rs', 'linestyle': '--', 'xlabel': "Angle d incidence (degrés)", 'ylabel': 'Reflectance / Transmittance', 'title': f"Tracé angulaire (λ = {res['l_a'][0]:.0f} nm)", 'cursor_format': "θ={x:.2f}°\n{label}={y:.3f}"},
        {'x': res['inc_a'], 'y': res['Rp_a'][0, :], 'label': 'Rp', 'linestyle': '--', 'xlabel': '', 'ylabel': '', 'title': '', 'cursor_format': "θ={x:.2f}°\n{label}={y:.3f}"},
        {'x': res['inc_a'], 'y': res['Ts_a'][0, :], 'label': 'Ts', 'linestyle': '-', 'xlabel': '', 'ylabel': '', 'title': '', 'cursor_format': "θ={x:.2f}°\n{label}={y:.3f}"},
        {'x': res['inc_a'], 'y': res['Tp_a'][0, :], 'label': 'Tp', 'linestyle': '-', 'xlabel': '', 'ylabel': '', 'title': '', 'cursor_format': "θ={x:.2f}°\n{label}={y:.3f}"}
    ]

    lines = []
    for i, trace in enumerate(traces):
        ax = axes[0, 0] if i < 4 else axes[0, 1]
        line = ax.plot(trace['x'], trace['y'], label=trace['label'], linestyle=trace['linestyle'])[0]
        lines.append(line)
        ax.set_xlabel(trace['xlabel'])
        ax.set_ylabel(trace['ylabel'])

        if i < 4:
            ax.set_title(f"Tracé spectral (incidence {inc:.1f}°)")
            # Ajout du quadrillage pour le tracé spectral
            ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
            ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
            ax.minorticks_on()
            ax.set_ylim(bottom=0)
            ax.set_ylim(top=1)
            # Ajustement de l'axe x
            ax.set_xlim(trace['x'][0], trace['x'][-1])

        else:
            ax.set_title(f"Tracé angulaire (λ = {res['l_a'][0]:.0f} nm)")
            # Ajout du quadrillage pour le tracé angulaire
            ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
            ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
            ax.minorticks_on()
            ax.set_ylim(bottom=0)
            ax.set_ylim(top=1)
            # Ajustement de l'axe x
            ax.set_xlim(trace['x'][0], trace['x'][-1])
        if i%4 == 3:
            ax.legend()

    def format_annotation(sel):
        trace_index = lines.index(sel.artist)
        sel.annotation.set_text(traces[trace_index]['cursor_format'].format(x=sel.target[0], y=sel.target[1], label=traces[trace_index]['label']))

    mplcursors.cursor(lines[:4], hover=True).connect("add", format_annotation)
    mplcursors.cursor(lines[4:], hover=True).connect("add", format_annotation)

    indices = [nH_r - 1j * nH_i if i % 2 == 0 else nL_r - 1j * nL_i for i in range(len(emp_str.split(',')))]

    n_reel = [np.real(n) for n in indices[::-1]]
    x_coords = np.concatenate(([-50, -0, 0], ep_cum, [ep_cum[-1] + 1, ep_cum[-1] + 51]))
    y_coords = [np.real(nSub), np.real(nSub), np.real(nSub)] + n_reel + [1, 1]

    ax1 = axes[1, 0]
    ax1.plot(x_coords, y_coords, drawstyle='steps-pre')
    ax1.set_xlabel('Epaisseur cumulée (nm)')
    ax1.set_ylabel('Partie réelle de l\'indice')
    # Ajout du quadrillage pour l'indice en fonction de l'épaisseur
    ax1.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
    ax1.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()
    # Ajustement de l'axe x pour le graphe de l'indice
    ax1.set_xlim(-50, ep_cum[-1] + 50)

    # Ajout des étiquettes SUBSTRAT et AIR
    ax1.text(-25, 1.05, "SUBSTRAT", ha='center', va='center', fontsize=8)
    ax1.text(ep_cum[-1] + 25, 1.05, "AIR", ha='center', va='center', fontsize=8)

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'TMonitoring (λ = {lambda_monitoring:.0f} nm)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1)

    x_coords_inter = []
    for i in range(len(emp)):
        for k in range(1, n_inter + 1):
            x_coords_inter.append(ep_cum[i-1] + ep[i] * k / (n_inter + 1) if i>0 else ep[i] * k / (n_inter + 1))

    x_coords_all = np.concatenate(([-50, 0], x_coords_inter, ep_cum))

    # Calcul de la transmission du substrat nu
    theta_inc_spectral = 0.
    alpha = 0.
    etainc = 1.
    etasub = nSub

    theta_inc_spectral = np.radians(inc)
    alpha = np.sin(theta_inc_spectral)
    etainc = np.sqrt(1. - alpha**2)
    etasub = np.sqrt(nSub**2 - alpha**2)

    if substrat_fini_var.get():
        # Substrat fini : Formule complète de la transmission, incluant les deux interfaces
        r_as = (etainc - etasub) / (etainc + etasub)
        T_substrat = (1 - np.abs(r_as)**2) / (1 - np.abs(r_as)**2)  # Simplification: vaut (1-R)/(1-R) = 1 si on s'arrete là
        # On doit diviser par le dénominateur habituel avec r_infini remplacé par r_as
        T_substrat = (1 - np.abs(r_as)**2) / (1 - np.abs(r_as)**4) # Correctif: on considère l'aller retour dans la lame de verre

    else:
        # Substrat infini:  On prend la transmission d'une seule interface air-substrat.
        T_substrat = (np.real(etasub) / np.real(etainc)) * np.abs(2 * etainc / (etainc + etasub))**2



    # Construction des y_coords en tenant compte du substrat nu
    y_coords_all = np.concatenate(([T_substrat, T_substrat], res['transmissions_inter'], res['transmissions'][1:]))


    sorted_indices = np.argsort(x_coords_all)
    x_coords_all_sorted = x_coords_all[sorted_indices]
    y_coords_all_sorted = y_coords_all[sorted_indices]


# Création de x_coords_smooth pour la partie constante (-50 à 0)
    x_coords_smooth_neg = np.linspace(-50, 0, 50)  # 50 points pour la constance, par exemple
    y_coords_smooth_neg = np.full_like(x_coords_smooth_neg, T_substrat) # Valeur constante

    # Création de x_coords_smooth pour la partie variable (0 à la fin)
    x_coords_smooth_pos = np.linspace(0, x_coords_all_sorted.max(), 450) # 450 points (total = 500)
    spl = make_interp_spline(x_coords_all_sorted, y_coords_all_sorted, k=3)
    y_coords_smooth_pos = spl(x_coords_smooth_pos)

    #On supprime la valeur x=0 qu'on a en double
    x_coords_smooth_pos = x_coords_smooth_pos[1:]
    y_coords_smooth_pos = y_coords_smooth_pos[1:]

    # Concaténation des deux parties
    x_coords_smooth = np.concatenate((x_coords_smooth_neg, x_coords_smooth_pos))
    y_coords_smooth = np.concatenate((y_coords_smooth_neg, y_coords_smooth_pos))


    # Prolonger la courbe Tmonitoring
    x_coords_smooth = np.append(x_coords_smooth, [ep_cum[-1] + 50])
    y_coords_smooth = np.append(y_coords_smooth, [res['transmissions'][-1]])

    # Tracé de la transmittance en fonction de l'épaisseur cumulée
    line_transmittance, = ax2.plot(x_coords_smooth, y_coords_smooth, 'r-', linewidth=1)
    ax2.plot(x_coords_all_sorted, y_coords_all_sorted, 'ro', markersize=1)

    # Ajustement de l'axe x pour le graphe de TMonitoring
    ax2.set_xlim(-50, ep_cum[-1] + 50)


    # Ajout du curseur pour la courbe de transmittance
    cursor_transmittance = mplcursors.cursor(line_transmittance, hover=True)
    cursor_transmittance.connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Épaisseur cumulée={sel.target[0]:.2f} nm\nTransmittance={sel.target[1]:.3f}"
        )
    )

    c = ['blue' if i % 2 == 0 else 'red' for i in range(len(indices))]
    axes[1, 1].barh(range(len(indices) - 1, -1, -1), np.array(ep)[::-1], align='center', color=c)
    axes[1, 1].set_yticks(range(len(indices)))
    axes[1, 1].set_yticklabels([f"Couche {i + 1}\n(n={n:.2f})" for i, n in enumerate(indices)], fontsize=8)
    axes[1, 1].set_xlabel('Epaisseur (nm)')
    axes[1, 1].set_title('Empilement (substrat en bas)')

    fontsize = max(8, 16 - len(indices) // 3)
    
    for i, (e, ind) in enumerate(zip(np.array(ep)[::-1], indices)):
        axes[1, 1].text(e / 2, len(indices) - i - 1, f"{e:.1f} nm", va='center', ha='center', color='white', fontsize=fontsize)

    plt.tight_layout()
    return fig

def update_layers_count(*args):
    """Met à jour le nombre de couches affiché."""
    try:
        emp_str = entry_vars['emp_str'].get()
        num_layers = len(emp_str.split(','))
        layers_count_label.config(text=f"Nombre de couches : {num_layers}")
    except:
        layers_count_label.config(text=f"Nombre de couches : Erreur")

def lancer_calcul():
    """Fonction principale pour lancer le calcul, l'affichage et la sauvegarde des résultats."""
    try:
        values = {k: float(v.get()) if k not in ['emp_str','lambda_monitoring'] else v.get() for k, v in entry_vars.items()}
        nH = values['nH_r'] - 1j * values['nH_i']
        nL = values['nL_r'] - 1j * values['nL_i']
        n_inter = int(values['n_inter'])
        substrat_fini = substrat_fini_var.get()
        emp_str = values['emp_str']
        num_layers = len(emp_str.split(','))
        export_excel = export_excel_var.get()
        lambda_monitoring = float(values['lambda_monitoring'])

        res, ep = calcul_empilement(nH, nL, values['nSub'], values['l0'], emp_str,
                                    (values['l_range_deb'], values['l_range_fin']), values['l_step'],
                                    (values['a_range_deb'], values['a_range_fin']), values['a_step'], values['inc'], n_inter, substrat_fini, lambda_monitoring)

        fig = tracer_graphiques(res, ep, values['nH_r'], values['nH_i'], values['nL_r'], values['nL_i'],
                                values['nSub'], emp_str, values['inc'], n_inter, lambda_monitoring)

        # Ajout de la figure pour la transmission spectrale en fonction de l'épaisseur cumulée
        fig_transmission, ax_transmission = plt.subplots(figsize=(8, 6))
        fig_transmission.canvas.manager.set_window_title('Transmission Spectrale en fonction de l\'épaisseur cumulée')

        # Création d'une liste pour stocker les étiquettes
        labels = []
        label = ax_transmission.text(0.95, 0.95, '', transform=ax_transmission.transAxes, ha='right', va='top', fontsize=10,
                                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        # Tracé initial (vide)
        line_transmission, = ax_transmission.plot([], [], 'b-')
        ax_transmission.set_xlabel('Longueur d\'onde (nm)')
        ax_transmission.set_ylabel('Transmission')
        ax_transmission.set_title('Transmission Spectrale en fonction de l\'épaisseur cumulée')
        ax_transmission.grid(True)
        ax_transmission.set_xlim(values['l_range_deb'], values['l_range_fin'])
        ax_transmission.set_ylim(0, 1)

        # Calcul des épaisseurs cumulées intermédiaires
        ep_cum = np.cumsum(ep)
        x_coords_inter = []
        for i in range(len(ep)):
            for k in range(1, n_inter + 1):
                x_coords_inter.append(ep_cum[i - 1] + ep[i] * k / (n_inter + 1) if i > 0 else ep[i] * k / (n_inter + 1))
        x_coords_all = np.concatenate(([0], x_coords_inter, ep_cum))
        x_coords_all_sorted = np.sort(x_coords_all)

        # Initialisation de l'animation
        def init_animation():
            line_transmission.set_data([], [])
            label.set_text('')
            return line_transmission, label

        # Fonction d'animation
        def animate_transmission(i, ep, res, line, label):
            current_ep_cum = x_coords_all_sorted[i]
            Ts_values = []

            # Trouver le numéro de la couche
            couche_num = 0
            ep_cumulee = 0
            for j, epaisseur in enumerate(ep):
                if current_ep_cum <= ep_cumulee + epaisseur:
                    couche_num = j + 1
                    break
                ep_cumulee += epaisseur
            # Déterminer l'épaisseur de la couche actuelle
            if couche_num == 1:
                epaisseur_couche = current_ep_cum
            else:
                epaisseur_couche = current_ep_cum - ep_cum[couche_num - 2]

            # Mettre à jour l'étiquette en temps réel
            label_text = f"Couche {couche_num}\n{epaisseur_couche:.2f} nm"
            label.set_text(label_text)

            for l_idx, l in enumerate(res['l']):
                M = np.eye(2, dtype=complex)
                theta_inc_spectral = np.radians(values['inc'])
                alpha = np.sin(theta_inc_spectral)
                etainc = np.sqrt(1. - alpha**2)
                etasub = np.sqrt(values['nSub']**2 - alpha**2)

                for index_couche, epaisseur_couche_totale in enumerate(ep):
                    Ni = nH if index_couche % 2 == 0 else nL
                    eta = np.sqrt(Ni**2 - alpha**2)

                    if current_ep_cum <= np.sum(ep[:index_couche]):
                        pass
                    elif current_ep_cum > np.sum(ep[:index_couche]) and current_ep_cum <= np.sum(ep[:index_couche + 1]):
                        epaisseur_courante = current_ep_cum - np.sum(ep[:index_couche])
                        phi = (2 * np.pi / l) * (eta * epaisseur_courante)
                        M = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                        [1j * eta * np.sin(phi), np.cos(phi)]]) @ M
                        break
                    elif current_ep_cum > np.sum(ep[:index_couche + 1]):
                        phi = (2 * np.pi / l) * (eta * epaisseur_couche_totale)
                        M = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                        [1j * eta * np.sin(phi), np.cos(phi)]]) @ M

                ts_infini = 2 * etainc / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0])
                Ts_infini = (np.real(np.abs(etasub)) / np.real(np.abs(etainc))) * np.abs(ts_infini)**2

                if substrat_fini:
                    Rb = np.abs((etainc - etasub) / (etainc + etasub))**2
                    T = ((1 - Rb) * Ts_infini) / (1 - np.abs((etainc * M[0, 0] - etasub * M[1, 1] + etainc * etasub * M[0, 1] - M[1, 0]) / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0]))**2 * Rb)
                else:
                    T = Ts_infini

                Ts_values.append(T)

            line.set_data(res['l'], Ts_values)
            return line, label

        # Création de l'animation
        ani = FuncAnimation(fig_transmission, animate_transmission, frames=len(x_coords_all_sorted),
                            fargs=(ep, res, line_transmission, label), init_func=init_animation, blit=True, repeat=False, interval=10)

        plt.show()

        # Enregistrement des résultats dans un fichier Excel (si demandé)
        if export_excel:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M-%S") # Format AAAA-MM-JJ-HH-MM-SS
            excel_file = f"Resultats_empilement_{num_layers}_couches_{timestamp}.xlsx"

            if os.path.exists(excel_file):
                os.remove(excel_file)

            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                # Onglet 1: Paramètres
                parameters = pd.DataFrame.from_dict(values, orient='index', columns=['Valeur'])
                parameters.loc['Substrat Fini'] = substrat_fini
                parameters.to_excel(writer, sheet_name='Paramètres')

                # Onglet 2: Données spectrales
                spectral_data = pd.DataFrame({
                    'Longueur d\'onde (nm)': res['l'],
                    'Rs': res['Rs_s'][:, 0],
                    'Rp': res['Rp_s'][:, 0],
                    'Ts': res['Ts_s'][:, 0],
                    'Tp': res['Tp_s'][:, 0]
                })
                spectral_data.to_excel(writer, sheet_name='Données Spectrales', index=False)

                # Onglet 3: Données angulaires
                angular_data = pd.DataFrame({
                    'Angle (°)': res['inc_a'],
                    'Rs': res['Rs_a'][0, :],
                    'Rp': res['Rp_a'][0, :],
                    'Ts': res['Ts_a'][0, :],
                    'Tp': res['Tp_a'][0, :]
                })
                angular_data.to_excel(writer, sheet_name='Données Angulaires', index=False)

                # Onglet 4: Monitoring
                # **Correction ici : Calculer ep_cum et x_coords_inter en utilisant la variable ep définie plus haut.**
                ep_cum = np.cumsum(ep)
                x_coords_inter = []
                for i in range(len(ep)):
                    for k in range(1, n_inter + 1):
                        x_coords_inter.append(ep_cum[i - 1] + ep[i] * k / (n_inter + 1) if i > 0 else ep[i] * k / (n_inter + 1))

                if substrat_fini:
                    x_coords_all = np.concatenate(([0], x_coords_inter, ep_cum))
                    y_coords_all = np.concatenate(([res['transmissions'][0]], res['transmissions_inter'], res['transmissions'][1:]))
                else:
                    x_coords_all = np.concatenate((x_coords_inter, ep_cum))
                    y_coords_all = np.concatenate((res['transmissions_inter'], res['transmissions'][1:]))

                transmissions_inter_data = pd.DataFrame({
                    'x_coords_all': x_coords_all,
                    'Transmission': y_coords_all
                })
                transmissions_inter_data.to_excel(writer, sheet_name='Monitoring', index=False)

                # Onglet 5: Transmissions aux interfaces
                transmissions_data = pd.DataFrame({
                    'Interface': range(len(res['transmissions'])),
                    'Transmission': res['transmissions']
                })
                transmissions_data.to_excel(writer, sheet_name='Transmissions Interfaces', index=False)

                # Ajustement automatique de la largeur des colonnes (exemple pour Données Spectrales)
                worksheet = writer.sheets['Données Spectrales']
                for i, col_data in enumerate(spectral_data):
                    max_len = spectral_data[col_data].astype(str).str.len().max()
                    worksheet.set_column(i, i, max_len + 2)

            messagebox.showinfo("Info", f"Résultats enregistrés dans {excel_file}")

    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")
    # Interface graphique
root = tk.Tk()
root.title("Calcul d'empilement de couches minces")

entry_vars = {}
entries = [("Matériau H (réel):", "2.25", 'nH_r'), ("Matériau H (imaginaire):", "0.0001", 'nH_i'),
            ("Matériau L (réel):", "1.48", 'nL_r'), ("Matériau L (imaginaire):", "0.0001", 'nL_i'),
            ("Substrat (indice):", "1.52", 'nSub'), ("Longueur d'onde de centrage (nm):", "550", 'l0'),
            ("Intervalle spectral début (nm):", "400", 'l_range_deb'),
            ("Intervalle spectral fin (nm):", "700", 'l_range_fin'),
            ("Intervalle angulaire début (degrés):", "0", 'a_range_deb'),
            ("Intervalle angulaire fin (degrés):", "89", 'a_range_fin'),
            ("Pas spectral (nm):", "1", 'l_step'), ("Pas angulaire (degrés):", "1", 'a_step'),
            ("Incidence (degrés):", "0", 'inc'), ("pts par couche monitoring:", "30", "n_inter"),
            ("Longueur d'onde monitoring (nm):", "550", 'lambda_monitoring')]

# Créer un frame pour les champs liés à l'empilement
emp_frame = ttk.Frame(root)
emp_frame.grid(row=7, column=0, columnspan=2, padx=0, pady=0, sticky="ew")

# Champ d'entrée pour l'empilement (QWOT)
ttk.Label(emp_frame, text="Empilement (QWOT):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
entry_vars['emp_str'] = ttk.Entry(emp_frame, width=36)
entry_vars['emp_str'].insert(0, "1,1,1,1,1,2,1,1,1,1,1")
entry_vars['emp_str'].bind("<KeyRelease>", update_layers_count)
entry_vars['emp_str'].grid(row=0, column=1, sticky="ew", padx=5, pady=5)

# Label pour afficher le nombre de couches
layers_count_label = ttk.Label(emp_frame, text="Nombre de couches : ", font=("Arial", 9, "bold"))
layers_count_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
update_layers_count()

# Placer les autres champs d'entrée en dehors du frame
for i, (label_text, default_value, var_name) in enumerate(entries):
    if var_name == 'emp_str':
        continue # Ignorer l'entrée de l'empilement car elle est déjà placée
    row_index = i if i < 7 else i + 1  # Ajuster l'index de ligne après l'empilement
    ttk.Label(root, text=label_text).grid(row=row_index, column=0, sticky="w", padx=5, pady=5)
    entry_vars[var_name] = ttk.Entry(root)
    entry_vars[var_name].insert(0, default_value)
    entry_vars[var_name].grid(row=row_index, column=1, padx=5, pady=5)

# Case à cocher pour le type de substrat
substrat_fini_var = tk.BooleanVar(value=False)
ttk.Checkbutton(root, text="Substrat fini", variable=substrat_fini_var).grid(row=len(entries) + 2, column=0, columnspan=2, pady=5, sticky="w")

# Case à cocher pour l'export Excel
export_excel_var = tk.BooleanVar(value=False)
ttk.Checkbutton(root, text="Exporter vers Excel", variable=export_excel_var).grid(row=len(entries) + 3, column=0, columnspan=2, pady=5, sticky="w")

# Bouton pour lancer le calcul
ttk.Button(root, text="Lancer le calcul", command=lancer_calcul).grid(row=len(entries) + 4, column=0, columnspan=2, pady=10)

def trace_rs_infini_complexe():
    """Ouvre une nouvelle fenêtre avec le tracé de rs_infini dans le plan complexe."""
    try:
        values = {k: float(v.get()) if k not in ['emp_str','lambda_monitoring'] else v.get() for k, v in entry_vars.items()}
        nH = values['nH_r'] - 1j * values['nH_i']
        nL = values['nL_r'] - 1j * values['nL_i']
        emp_str = values['emp_str']
        lambda_monitoring = float(values['lambda_monitoring'])
        n_inter = int(values['n_inter'])

        # Utilisation des mêmes valeurs que calcul_empilement
        res, ep = calcul_empilement(nH, nL, values['nSub'], values['l0'], emp_str,
                                    (values['l_range_deb'], values['l_range_fin']), values['l_step'],
                                    (values['a_range_deb'], values['a_range_fin']), values['a_step'], values['inc'], n_inter, substrat_fini_var.get(), lambda_monitoring)

        ep_cum = np.cumsum(ep)

        fig_complexe, ax_complexe = plt.subplots(figsize=(6, 6))
        fig_complexe.canvas.manager.set_window_title('rs_infini dans le plan complexe')

        # Calcul de rs_infini pour les points intermédiaires
        theta_inc_spectral = np.radians(values['inc'])
        alpha = np.sin(theta_inc_spectral)
        rs_infinis = []

        # Construction de x_coords_all qui contient toutes les épaisseurs,
        # SAUF le dernier point intermédiaire de chaque couche
        x_coords_inter = []
        for i in range(len(ep)):
            for k in range(1, n_inter + 1):  # k va de 1 à n_inter + 1
                x_coords_inter.append(ep_cum[i-1] + ep[i] * k / (n_inter + 1) if i > 0 else ep[i] * k / (n_inter + 1))
        x_coords_all = np.concatenate(([0], x_coords_inter, ep_cum))

        for ep_courante in x_coords_all:
            M = np.eye(2, dtype=complex)
            etainc = np.sqrt(1 - alpha ** 2)
            etasub = np.sqrt(values['nSub'] ** 2 - alpha ** 2)

            for index_couche, epaisseur_couche in enumerate(ep):
                Ni = nH if index_couche % 2 == 0 else nL
                eta = np.sqrt(Ni ** 2 - alpha ** 2)

                #Si on a déjà dépassé l'épaisseur cumulée courante
                if ep_courante <= np.sum(ep[:index_couche]):
                    pass #on ne fait rien

                #Si l'épaisseur cumulée courante est dans la couche actuelle
                elif ep_courante >= np.sum(ep[:index_couche]) and ep_courante <= np.sum(ep[:index_couche+1]):
                    #Cas où l'on est dans les points intermédiaires (strictement inf à l'epaisseur totale de la couche)
                    if ep_courante < np.sum(ep[:index_couche+1]):
                        phi = (2 * np.pi / lambda_monitoring) * (eta * (ep_courante - np.sum(ep[:index_couche])))
                        M = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                        [1j * eta * np.sin(phi), np.cos(phi)]]) @ M
                        break #On a traité l'épaisseur cumulée courante, on sort de la boucle

                    #Cas où l'on est à la fin de la couche
                    else:
                        phi = (2 * np.pi / lambda_monitoring) * (eta * epaisseur_couche)
                        M = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                        [1j * eta * np.sin(phi), np.cos(phi)]]) @ M

                #Si l'épaisseur cumulée courante n'est pas dans la couche actuelle, on prend l'épaisseur totale de la couche
                elif ep_courante > np.sum(ep[:index_couche+1]):
                    phi = (2 * np.pi / lambda_monitoring) * (eta * epaisseur_couche)
                    M = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                    [1j * eta * np.sin(phi), np.cos(phi)]]) @ M

            rs_infini = (etainc * M[0, 0] - etasub * M[1, 1] + etainc * etasub * M[0, 1] - M[1, 0]) / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0])
            rs_infinis.append(rs_infini)

        # Tracé dans le plan complexe
        colors = ['blue', 'red']  # Couleurs pour les couches paires et impaires

        point_handles = []  # On initialise une liste vide qui contiendra les points

        # Dictionnaire pour stocker l'index de la couche pour chaque point
        couche_indices = {}

        # Tracé des points individuels
        for i in range(len(x_coords_all)):
            couche_index = 0
            ep_cumul = 0
            for j, epaisseur in enumerate(ep):
                if x_coords_all[i] <= ep_cumul + epaisseur:
                    couche_index = j
                    break
                ep_cumul += epaisseur

            couche_indices[i] = couche_index
            color = colors[couche_index % 2]  # Alterner les couleurs

            point, = ax_complexe.plot(np.real(rs_infinis[i]), np.imag(rs_infinis[i]), marker='o', linestyle='none', markersize=1, color=color)
            point_handles.append(point)

            # Ajout de l'étiquette avec le numéro de couche (1 point sur 15)
            if i % 15 == 0:
                ax_complexe.annotate(str(couche_index+1), (np.real(rs_infinis[i]), np.imag(rs_infinis[i])),
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=8, color=color)

        # Lissage des points avec une interpolation spline
        x_coords_all_sorted = np.linspace(x_coords_all.min(), x_coords_all.max(), 300)

        # S'assurer que les données sont triées par ordre croissant pour l# S'assurer que les données sont triées par ordre croissant pour l'interpolation
        sort_indices = np.argsort(x_coords_all)
        x_coords_all_sorted_interp = x_coords_all[sort_indices]
        rs_infinis_sorted = np.array(rs_infinis)[sort_indices]

        # Création d'une interpolation spline pour les parties réelles et imaginaires
        spl_real = make_interp_spline(x_coords_all_sorted_interp, np.real(rs_infinis_sorted), k=3)
        spl_imag = make_interp_spline(x_coords_all_sorted_interp, np.imag(rs_infinis_sorted), k=3)

        # Création de la courbe lissée
        rs_smooth_real = spl_real(x_coords_all_sorted)
        rs_smooth_imag = spl_imag(x_coords_all_sorted)
        line_smooth, = ax_complexe.plot(rs_smooth_real, rs_smooth_imag, '-', color='black', linewidth=1)

        # Configuration du graphique
        ax_complexe.set_xlabel('Re(rs_infini)')
        ax_complexe.set_ylabel('Im(rs_infini)')
        ax_complexe.set_title('rs_infini dans le plan complexe')
        ax_complexe.grid(True)
        ax_complexe.set_aspect('equal')  # Assurer que les axes sont à la même échelle
        ax_complexe.axhline(0, color='black', linewidth=0.5) # Ligne horizontale pour l'axe réel
        ax_complexe.axvline(0, color='black', linewidth=0.5) # Ligne verticale pour l'axe imaginaire

        # Configuration du curseur pour afficher l'épaisseur et le numéro de couche
        cursor = mplcursors.cursor(point_handles, hover=True)

        def update_annotation(sel):
            index = point_handles.index(sel.artist)
            couche_index = couche_indices.get(index, 0)
            epaisseur = x_coords_all[index]
            sel.annotation.set_text(
                f'Couche: {couche_index + 1}\n'
                f'Épaisseur: {epaisseur:.2f} nm\n'
                f'Re: {np.real(rs_infinis[index]):.3f}\n'
                f'Im: {np.imag(rs_infinis[index]):.3f}'
            )
            sel.annotation.get_bbox_patch().set_facecolor('white')
            sel.annotation.get_bbox_patch().set_alpha(0.8)

        cursor.connect("add", update_annotation)
        plt.show()


    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

# Ajout du bouton pour le tracé dans le plan complexe
ttk.Button(root, text="Tracer rs_infini (complexe)", command=trace_rs_infini_complexe).grid(row=len(entries) + 5, column=0, columnspan=2, pady=10)


def lancer_monitoring_direct():
    """Ouvre une nouvelle fenêtre avec le tracé de la transmission spectrale simulée."""
    try:
        values = {k: float(v.get()) if k not in ['emp_str','lambda_monitoring'] else v.get() for k, v in entry_vars.items()}
        nH = values['nH_r'] - 1j * values['nH_i']
        nL = values['nL_r'] - 1j * values['nL_i']
        emp_str = values['emp_str']
        n_inter = int(values['n_inter'])
        lambda_monitoring = float(values['lambda_monitoring'])
        res, ep = calcul_empilement(nH, nL, values['nSub'], values['l0'], emp_str,
                                    (values['l_range_deb'], values['l_range_fin']), values['l_step'],
                                    (values['a_range_deb'], values['a_range_fin']), values['a_step'],
                                    values['inc'], n_inter, substrat_fini_var.get(),lambda_monitoring)
        ep_cum = np.cumsum(ep)

        fig_monitoring, ax_monitoring = plt.subplots(figsize=(8, 6))
        fig_monitoring.canvas.manager.set_window_title('Simulation du monitoring direct')

        # Initialisation des variables pour l'animation
        line_transmission, = ax_monitoring.plot([], [], 'b-', label='Transmission Spectrale')
        point, = ax_monitoring.plot([], [], 'ro') # Point rouge pour le monitoring

        # Définition des étiquettes et configuration initiale
        ax_monitoring.set_xlabel('Longueur d\'onde (nm)')
        ax_monitoring.set_ylabel('Transmission')
        ax_monitoring.set_title('Simulation du monitoring direct')
        ax_monitoring.grid(True)
        ax_monitoring.legend()
        ax_monitoring.set_xlim(values['l_range_deb'], values['l_range_fin'])
        ax_monitoring.set_ylim(0, 1)

        # Label pour afficher l'épaisseur cumulée
        label_epaisseur = ax_monitoring.text(0.05, 0.95, '', transform=ax_monitoring.transAxes, ha='left', va='top',
                                             fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        # Configuration du curseur pour la courbe de transmission
        cursor_transmission = mplcursors.cursor(line_transmission, hover=True)

        def update_annotation(sel):
             sel.annotation.set_text(f"λ = {sel.target[0]:.2f} nm\nT = {sel.target[1]:.3f}")

        cursor_transmission.connect("add", update_annotation)

        # Création d'une liste x_coords_all complète pour l'animation
        x_coords_inter = []
        for i in range(len(ep)):
            for k in range(1, n_inter + 1):
                x_coords_inter.append(ep_cum[i - 1] + ep[i] * k / (n_inter + 1) if i > 0 else ep[i] * k / (n_inter + 1))

        x_coords_all = np.concatenate(([0], x_coords_inter, ep_cum))

        # Initialisation de l'animation
        def init_animation():
            line_transmission.set_data([], [])
            point.set_data([], [])  # Initialiser le point
            label_epaisseur.set_text('') #initialiser l'épaisseur
            return line_transmission, point, label_epaisseur,

        # Fonction d'animation
        def animate(i):
            current_ep_cum = x_coords_all[i]
            Ts_values = []
            label_epaisseur.set_text(f'Épaisseur cumulée = {current_ep_cum:.2f} nm')

            for l_idx, l in enumerate(res['l']):
                # Calcul de la matrice M pour chaque longueur d'onde
                M = np.eye(2, dtype=complex)
                theta_inc_spectral = np.radians(values['inc'])
                alpha = np.sin(theta_inc_spectral)
                etainc = np.sqrt(1. - alpha**2)
                etasub = np.sqrt(values['nSub']**2 - alpha**2)

                for index_couche, epaisseur_couche in enumerate(ep):
                    Ni = nH if index_couche % 2 == 0 else nL
                    eta = np.sqrt(Ni**2 - alpha**2)

                    if current_ep_cum <= np.sum(ep[:index_couche]):
                        pass
                    elif current_ep_cum >= np.sum(ep[:index_couche]) and current_ep_cum <= np.sum(ep[:index_couche + 1]):
                        epaisseur_courante = current_ep_cum - np.sum(ep[:index_couche])
                        phi = (2 * np.pi / l) * (eta * epaisseur_courante)
                        M = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                    [1j * eta * np.sin(phi), np.cos(phi)]]) @ M
                        break #on a traité l'épaisseur, pas besoin de regarder les couches suivantes
                    elif current_ep_cum > np.sum(ep[:index_couche + 1]):
                        phi = (2 * np.pi / l) * (eta * epaisseur_couche)
                        M = np.array([[np.cos(phi), (1j / eta) * np.sin(phi)],
                                    [1j * eta * np.sin(phi), np.cos(phi)]]) @ M

                ts_infini = 2 * etainc / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0])
                Ts_infini = (np.real(np.abs(etasub)) / np.real(np.abs(etainc))) * np.abs(ts_infini)**2

                if substrat_fini_var.get():
                    Rb = np.abs((etainc - etasub) / (etainc + etasub))**2
                    T = ((1 - Rb) * Ts_infini) / (1 - np.abs((etainc * M[0, 0] - etasub * M[1, 1] + etainc * etasub * M[0, 1] - M[1, 0]) / (etainc * M[0, 0] + etasub * M[1, 1] + etainc * etasub * M[0, 1] + M[1, 0])) ** 2 * Rb)
                else:
                    T = Ts_infini

                Ts_values.append(T)

            # Trouver la valeur de T correspondant à lambda_monitoring
            t_monitoring = None
            for l_idx, l in enumerate(res['l']):
                if abs(l - lambda_monitoring) < 1e-6:  # Comparaison avec une petite tolérance
                    t_monitoring = Ts_values[l_idx]
                    break

            # Mettre à jour le point de monitoring
            if t_monitoring is not None:
                point.set_data([lambda_monitoring], [t_monitoring])

            line_transmission.set_data(res['l'], Ts_values)
            return line_transmission, point, label_epaisseur,

        ani = FuncAnimation(fig_monitoring, animate, frames=len(x_coords_all),
                        init_func=init_animation, blit=True, repeat=False, interval = 10)

        plt.show()

    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

ttk.Button(root, text="Lancer le monitoring direct", command=lancer_monitoring_direct).grid(row=len(entries) + 6, column=0, columnspan=2, pady=10)

root.mainloop()
