# torch_portfolio_report_dynamic.py
# Extended: facility static-pressure sweep, flowing-before-ignition Mach cases,
# hot-side heat-flux ramp, optional radiation, conservative MAWP options, and
# auto-updating plot labels.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
from pathlib import Path

# ==============================
# SUPER SIMPLE KNOBS (edit these)
# ==============================
SIMPLE_INPUTS = dict(
    # Base supply cases (stagnation supplies). Add/modify as needed.
    cases=[
        {"label": "A", "P0_O2_psig": 30.0, "P0_H2_psig": 46.0, "T0": 300.0, "p_ann_psia": 29.4},
        {"label": "B", "P0_O2_psig": 60.0, "P0_H2_psig": 90.0, "T0": 300.0, "p_ann_psia": 51.0},
        # {"label": "C", "P0_O2_psig": 60.0, "P0_H2_psig": 120.0, "T0": 300.0, "p_ann_psia": 60.0},
    ],

    # Facility scenarios (auto-expands each base case into scenario cases)
    use_facility_scenarios=True,
    no_flow_psia_grid=[4.0, 6.0, 8.0, 10.0, 12.0, 14.7],   # ejector/static sweep (psia)
    flowing_M_list=[0.8, 0.9, 1.0],                        # injector exit Mach for P/Pt

    # Transient thermal runtime and hot-side ramp (seconds)
    t_end=1.0,
    t_ramp_hot=0.10,                # time to ramp hot-side heat flux 0→100%
    ramp_shape="linear",           # "linear" (default) or "s-curve"

    # Hot-side & near-wall tuning knobs
    h_hot_ref=1.0e4,                # W/m^2-K @ reference mdot (first case)
    film_factor=0.90,               # near-wall gas temp ≈ film_factor * Tad

    # Radiation (outer wall to ambient). Set eps_rad=0 to disable.
    eps_rad=0.20,                   # emissivity
    T_rad_inf=300.0,                # ambient radiation sink temperature [K]

    # Output directory
    outdir=Path("./torch_report_out"),
)

# ==============================
# Detailed configuration (rarely change)
# ==============================
cfg = dict(
    # Geometry / material
    L_cooled = 6.0*0.0254,         # m
    D_core   = 4.2545e-3,          # m
    t_wall   = 0.254e-3,           # m
    gap_ann  = 0.4255e-3,          # m
    rho_s    = 8000.0,             # kg/m3
    cp_s     = 500.0,              # J/kg-K
    k_s      = 16.0,               # W/m-K
    T_allow  = 1000.0,             # K (1 s)
    T_H2_in  = 300.0,              # K

    # Chemistry / near-wall
    film_factor = 0.90,
    T_gas_film_fallback = 0.9*3300.0,

    # Orifices & discharge
    d_O2_core = 3.175e-3,
    d_H2_core = 0.8382e-3,
    d_H2_ann  = 1.8e-3,
    Cd_O2     = 0.8,
    Cd_H2_core= 0.8,
    Cd_H2_ann = 0.6,

    # Gas constants (ideal fallback)
    R_O2 = 259.8,     # J/kg-K
    R_H2 = 4124.0,    # J/kg-K
    gam_O2 = 1.4,
    gam_H2 = 1.4,

    # Coolant property fallbacks (if no REFPROP/CoolProp)
    mu_H2_fallback = 9.0e-6,  # Pa-s @ ~300K
    k_H2_fallback  = 0.19,    # W/m-K
    cp_H2_fallback = 14300.0, # J/kg-K

    # Pc model (used if scenario doesn’t override)
    Pc_model = "exhaust_choked",        # {"annulus_psia", "exhaust_choked", "override"}
    Pc_override_psia = None,             # set a float to force Pc

    # Ignition model (placeholder; ramp helps approximate build-up)
    tau_ign_guess = 1.0e-4,    # s
    Lchar_model = "D_core",    # {"D_core", "recirc_simple"}
    recirc_k = 6.0,            # L_char = k * D_core if "recirc_simple"

    # Monte Carlo
    N_mc = 600,
    sigma_h_hot = 0.15,
    sigma_h_cool = 0.10,
    sigma_Tgas = 0.05,
    sigma_k_s = 0.05,
    sigma_cp_s = 0.05,
)

# ==============================
# Structural tuning (optional)
# ==============================
STRUCT = dict(
    weld_eff=1.0,     # joint efficiency (≤1.0); set <1.0 for welded seams
    sf_allow=1.0,     # safety factor applied to allowable stress when computing MAWP
)

# ==============================
# Optional dependencies
# ==============================
HAVE_CT = False
try:
    import cantera as ct
    HAVE_CT = True
except Exception:
    HAVE_CT = False

# (Transport via REFPROP/CoolProp omitted here for brevity; we keep simple fallbacks.)

# ==============================
# Helpers
# ==============================
PSI2PA = 6894.75729
AMBIENT_PSIA = 14.7
SIGMA = 5.670374419e-8  # Stefan-Boltzmann

def psig_to_Pa(psig):
    return (psig + AMBIENT_PSIA)*PSI2PA

def psia_to_Pa(psia):
    return (psia)*PSI2PA


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def format_case_name(label: str, P0_O2_psig: float, P0_H2_psig: float, T0: float) -> str:
    return f"{label} — O2 {P0_O2_psig:.0f} psig / H2 {P0_H2_psig:.0f} psig @ {T0:.0f} K"


def build_cases(simple_cases):
    cases = []
    for c in simple_cases:
        name = format_case_name(c.get("label", "Case"), c["P0_O2_psig"], c["P0_H2_psig"], c["T0"])
        cases.append(dict(
            name=name,
            P0_O2_psig=float(c["P0_O2_psig"]),
            P0_H2_psig=float(c["P0_H2_psig"]),
            T0=float(c["T0"]),
            p_ann_psia=float(c["p_ann_psia"]),
        ))
    return cases


def isentropic_P_over_Pt(gamma, M):
    return (1.0 + 0.5*(gamma-1.0)*M*M)**(-gamma/(gamma-1.0))


def expand_facility_cases(base_cases, simple_inputs, gamma_mix_guess=1.30):
    """Expand each base case into additional 'no-flow' and 'flowing' subcases with overridden Pc.
    - no-flow: set Pc directly to specified static psia values (ejector sweep)
    - flowing: Pc = (P/Pt)(M,gamma)*Pt_mix, with Pt_mix mass-flow-weighted from supplies
    Returns a list of case dicts with optional keys: Pc_override_pa and label suffixes.
    """
    cases_out = []
    for base in base_cases:
        # Always include the original base case
        cases_out.append(dict(**base))
        if not SIMPLE_INPUTS.get('use_facility_scenarios', True):
            continue
        for pstat in SIMPLE_INPUTS.get('no_flow_psia_grid', []):
            c = dict(**base)
            c['name'] = f"{base['name']} — NoFlow Ps={pstat:.1f} psia"
            c['Pc_override_pa'] = psia_to_Pa(pstat)
            c['facility_mode'] = 'no_flow'
            cases_out.append(c)
        for M in SIMPLE_INPUTS.get('flowing_M_list', []):
            frac = isentropic_P_over_Pt(gamma_mix_guess, M)
            c = dict(**base)
            c['name'] = f"{base['name']} — Flow M={M:.1f} (P/Pt={frac:.2f})"
            c['flow_M'] = float(M)
            c['facility_mode'] = 'flowing'
            cases_out.append(c)
    return cases_out


def choked_mdot(Cd,A,P0,T0,gam,R):
    crit = (2.0/(gam+1.0))**((gam+1.0)/(2.0*(gam-1.0)))
    return Cd*A*P0*m.sqrt(gam/(R*T0)*crit)


def annulus_h_props(mdot_H2_ann, p_ann_Pa, T_ann, Dh_ann, A_ann_flow, mu, k, cp):
    rho = p_ann_Pa/(cfg['R_H2']*T_ann)
    U   = mdot_H2_ann/(rho*A_ann_flow + 1e-16)
    Re  = rho*U*Dh_ann/(mu + 1e-16)
    Pr  = cp*mu/(k + 1e-16)
    Nu  = 0.023*(Re**0.8)*(Pr**0.4)
    return Nu*k/(Dh_ann + 1e-16), Re, U, rho, Pr


def cantera_Tad_phi(phi, P, T0=300.0, diluent=None):
    """Return adiabatic flame temperature Tad and products mix at constant pressure, H2/O2 (+optional N2)."""
    if not HAVE_CT:
        return cfg['T_gas_film_fallback'] / cfg['film_factor'], None
    try:
        gas = ct.Solution('h2o2.yaml')
    except Exception:
        return cfg['T_gas_film_fallback'] / cfg['film_factor'], None
    comp = f'H2:{phi}, O2:1.0'
    if diluent:
        comp += f', {diluent}'
    gas.TPX = T0, P, comp
    gas.equilibrate('HP')
    return gas.T, gas


def cantera_mix_props(gas):
    if not HAVE_CT or gas is None:
        return 1.30, 300.0
    gamma = gas.cp_mass/gas.cv_mass
    Rm = ct.gas_constant/gas.mean_molecular_weight
    return gamma, Rm

# ==============================
# Transient two-node with hot-side ramp and optional outer-wall radiation
# ==============================

def simulate_two_node_ramped(h_hot_base, h_cool, A_inner, A_outer, A_cond, t_wall,
                              T_gas, T_H2, rho_s, cp_s, k_s,
                              t_end=1.0, dt=1e-3,
                              t_ramp=0.1, ramp_shape="linear",
                              eps_rad=0.0, T_rad_inf=300.0):
    """Two-node transient with a time-varying hot-side h (ramp) and optional outer-wall radiation.
    Inner node ≈ inner half-thickness; outer node ≈ outer half-thickness.
    Radiation is linearized each step: h_rad_outer ≈ 4·eps·σ·T_outer^3.
    """
    C_half = rho_s*cp_s*(A_cond * t_wall/2.0)
    Rcond  = (t_wall/2.0)/(k_s*A_cond + 1e-16)
    # Constant cool-side film coefficient as input; radiation added per-step
    Rhot0  = 1.0/(h_hot_base*A_inner + 1e-16)  # base (will scale with ramp)

    N = int(t_end/dt)
    t = np.linspace(0, t_end, N+1)
    T = np.array([300.0, 300.0], dtype=float)
    Ti = np.zeros(N+1); To = np.zeros(N+1)
    Ti[0]=T[0]; To[0]=T[1]

    def ramp_factor(ti):
        if t_ramp <= 0.0:
            return 1.0
        x = max(0.0, min(1.0, ti/t_ramp))
        if ramp_shape == "s-curve":
            # Smoothstep 3x^2 - 2x^3
            return (3*x*x - 2*x*x*x)
        return x  # linear

    for n in range(N):
        # Effective hot-side h at this time
        h_hot_t = h_hot_base * ramp_factor(t[n])
        Rhot = 1.0/(h_hot_t*A_inner + 1e-16)
        # Outer-wall radiation (linearize around current outer-wall temp)
        h_rad_outer = 4.0*eps_rad*SIGMA*max(To[n], 300.0)**3  # W/m^2-K
        Rcool_eff = 1.0/((h_cool + h_rad_outer)*A_outer + 1e-16)

        # Build per-step matrices
        C = np.array([C_half, C_half])
        G = np.array([[ 1.0/Rhot + 1.0/Rcond,  -1.0/Rcond           ],
                      [ -1.0/Rcond,            1.0/Rcond + 1.0/Rcool_eff]])
        S = np.array([ T_gas/Rhot, T_H2/Rcool_eff ])
        A = np.diag(C/dt) + G
        rhs = (C/dt)*T + S
        T = np.linalg.solve(A, rhs)
        Ti[n+1]=T[0]; To[n+1]=T[1]

    return t, Ti, To


def wall_heat_flux_from_h(h, T_gas, T_wall):
    return h * (T_gas - T_wall)


def hoop_stress_thinwall(Pc, D_i, t):
    return Pc * D_i / (2.0 * t + 1e-16)


def sigma_yield_316L(T):
    if T <= 300:  return 205e6
    if T >= 1000: return 100e6
    return 205e6 - (105e6)*(T-300.0)/(700.0)


def yield_FoS_at_T(Pc, D_i, t, T):
    sig_hoop = hoop_stress_thinwall(Pc, D_i, t)
    sig_y    = sigma_yield_316L(T)
    return sig_y / max(sig_hoop, 1e-9), sig_hoop, sig_y


def MAWP_from_yield(sig_allow, D_i, t):
    return 2.0 * t * sig_allow / (D_i + 1e-16)

# ==============================
# Main computation
# ==============================

def run(full_cfg, simple_inputs):
    # Merge simple knobs into cfg
    full_cfg = dict(full_cfg)
    # Mirror selected knobs
    if 'film_factor' in simple_inputs:
        full_cfg['film_factor'] = simple_inputs['film_factor']
    full_cfg['cases'] = build_cases(simple_inputs['cases'])
    full_cfg['cases'] = expand_facility_cases(full_cfg['cases'], simple_inputs)
    full_cfg['t_end'] = float(simple_inputs.get('t_end', 1.0))
    full_cfg['t_ramp_hot'] = float(simple_inputs.get('t_ramp_hot', 0.10))
    full_cfg['ramp_shape'] = str(simple_inputs.get('ramp_shape', 'linear'))
    full_cfg['eps_rad'] = float(simple_inputs.get('eps_rad', 0.0))
    full_cfg['T_rad_inf'] = float(simple_inputs.get('T_rad_inf', 300.0))
    full_cfg['outdir'] = simple_inputs.get('outdir', Path('./torch_report_out'))

    ensure_outdir(full_cfg['outdir'])

    # Geometry & areas
    L = full_cfg['L_cooled']; D_core = full_cfg['D_core']; t_wall = full_cfg['t_wall']; gap = full_cfg['gap_ann']
    D_OD = D_core + 2*t_wall
    A_inner = m.pi*D_core*L
    A_outer = m.pi*D_OD*L
    A_cond  = m.pi*((D_core+D_OD)/2.0)*L
    D_mean_ann = D_OD + 0.5*gap
    Dh_ann = 2.0*gap
    A_ann_flow = m.pi*D_mean_ann*gap

    A_O2 = m.pi*(full_cfg['d_O2_core']*0.5)**2
    A_H2c= m.pi*(full_cfg['d_H2_core']*0.5)**2
    A_H2a= m.pi*(full_cfg['d_H2_ann'] *0.5)**2

    # Loop cases
    flow_rows = []
    thermal_rows = []
    ign_rows = []
    energy_rows = []

    mdots_total = []
    h_hot_ref = None
    h_hot_ref_base = float(simple_inputs.get('h_hot_ref', 1.0e4))
    mref = None

    for case in full_cfg['cases']:
        P0_O2 = psig_to_Pa(case['P0_O2_psig'])
        P0_H2 = psig_to_Pa(case['P0_H2_psig'])
        T0    = case['T0']
        p_ann = psia_to_Pa(case['p_ann_psia'])

        mdO2 = choked_mdot(full_cfg['Cd_O2'], A_O2, P0_O2, T0, full_cfg['gam_O2'], full_cfg['R_O2'])
        mdH2c= choked_mdot(full_cfg['Cd_H2_core'], A_H2c, P0_H2, T0, full_cfg['gam_H2'], full_cfg['R_H2'])
        mdH2a= choked_mdot(full_cfg['Cd_H2_ann'],  A_H2a, P0_H2, T0, full_cfg['gam_H2'], full_cfg['R_H2'])
        mdH2 = mdH2c + mdH2a
        mdots_total.append(mdO2 + mdH2)

        OF   = mdO2 / (mdH2 + 1e-16)
        phi  = (1.0/OF) / (1.0/8.0)

        # Cantera: Tad and mix
        Tad, gas_prod = cantera_Tad_phi(phi, P0_O2, T0=T0)
        T_gas_film = full_cfg['film_factor']*Tad if HAVE_CT else full_cfg['T_gas_film_fallback']
        gamma_mix, R_mix = cantera_mix_props(gas_prod)

        # Transport for H2 coolant (annulus) at (T0, p_ann)
        mu_H2 = full_cfg['mu_H2_fallback']; k_H2=full_cfg['k_H2_fallback']; cp_H2=full_cfg['cp_H2_fallback']
        hH2, Re, U_ann, rhoH2, Pr = annulus_h_props(mdH2a, p_ann, T0, Dh_ann, A_ann_flow, mu_H2, k_H2, cp_H2)

        # Hot-side h: scale with total mdot^0.8; anchor at first case (but ramp will apply in-time)
        if h_hot_ref is None:
            h_hot_ref = h_hot_ref_base
            mref = mdots_total[-1]
        h_hot_base = h_hot_ref * ((mdots_total[-1]/mref) ** 0.8)

        # --- Pc models (facility overrides first) ---
        Pc_model = full_cfg['Pc_model']
        Pc_used = None
        if 'Pc_override_pa' in case:
            Pc_used = case['Pc_override_pa']
        elif case.get('facility_mode') == 'flowing':
            # mixed stagnation from supplies, mass-flow-weighted (rough)
            Pt_mix = (mdO2*P0_O2 + mdH2*P0_H2) / max((mdO2+mdH2), 1e-12)
            frac = isentropic_P_over_Pt(gamma_mix, case.get('flow_M', 0.9))
            Pc_used = max(frac * Pt_mix, AMBIENT_PSIA*PSI2PA)
        else:
            if Pc_model == "override" and full_cfg['Pc_override_psia'] is not None:
                Pc_used = psia_to_Pa(full_cfg['Pc_override_psia'])
            elif Pc_model == "annulus_psia":
                Pc_used = max(case['p_ann_psia'], AMBIENT_PSIA)*PSI2PA
            elif Pc_model == "exhaust_choked":
                A_throat = m.pi*(D_core*0.5)**2
                crit = (2.0/(gamma_mix+1.0))**((gamma_mix+1.0)/(2.0*(gamma_mix-1.0)))
                Pt = (mdots_total[-1]) / (A_throat * m.sqrt(gamma_mix/(R_mix*T_gas_film)*crit) + 1e-16)
                Pc_used = max(Pt * (2.0/(gamma_mix+1.0))**(gamma_mix/(gamma_mix-1.0)), AMBIENT_PSIA*PSI2PA)
            else:
                Pc_used = max(case['p_ann_psia'], AMBIENT_PSIA)*PSI2PA

        # Two-node transient with ramp and radiation
        t, Ti, To = simulate_two_node_ramped(
            h_hot_base, hH2, A_inner, A_outer, A_cond, t_wall,
            T_gas_film, full_cfg['T_H2_in'], full_cfg['rho_s'], full_cfg['cp_s'], full_cfg['k_s'],
            t_end=full_cfg['t_end'], dt=1e-3,
            t_ramp=full_cfg['t_ramp_hot'], ramp_shape=full_cfg['ramp_shape'],
            eps_rad=full_cfg['eps_rad'], T_rad_inf=full_cfg['T_rad_inf']
        )

        # Energy sanity (using steady h_hot_base at end-wall temp — conservative)
        qpp  = wall_heat_flux_from_h(h_hot_base, T_gas_film, Ti[-1])
        Qsurf= qpp*A_inner
        eta_chem = 0.45
        LHV_H2 = 120e6
        Qavail= eta_chem * mdH2 * LHV_H2

        # Structural
        FoS, sig_hoop, sig_y = yield_FoS_at_T(Pc_used, D_core, t_wall, Ti[-1])
        sig_allow = STRUCT['weld_eff'] * (sig_y / max(STRUCT['sf_allow'], 1e-9))
        Pc_allow = MAWP_from_yield(sig_allow, D_core, t_wall)

        # Pack rows
        flow_rows.append(dict(case=case['name'], mdot_O2_g_s=mdO2*1e3, mdot_H2_g_s=mdH2*1e3,
                              mdot_H2_ann_g_s=mdH2a*1e3, O_F=OF, phi=phi,
                              P0_O2_psig=case['P0_O2_psig'], P0_H2_psig=case['P0_H2_psig']))
        thermal_rows.append(dict(case=case['name'], T_inner_end=Ti[-1], T_outer_end=To[-1],
                                 h_hot_base=h_hot_base, h_H2=hH2, U_ann=U_ann, Re_ann=Re, Pr_ann=Pr,
                                 T_gas_film=T_gas_film, gamma_mix=gamma_mix, R_mix=R_mix, Pc_used=Pc_used))
        energy_rows.append(dict(case=case['name'], Qsurf_W=Qsurf, Qavail_W=Qavail, margin_W=Qavail-Qsurf))
        # Ignition proxy (Damkohler w/ ramped start is tricky — report recirc metric & U)
        L_char = (full_cfg['recirc_k']*D_core) if (full_cfg['Lchar_model']=="recirc_simple") else D_core
        ign_rows.append(dict(case=case['name'], L_char_m=L_char, U_ann=U_ann, note="ramp applied"))

    # DataFrames
    df_flows = pd.DataFrame(flow_rows)
    df_thermal = pd.DataFrame(thermal_rows)
    df_energy = pd.DataFrame(energy_rows)
    df_ign = pd.DataFrame(ign_rows)

    # Margins
    rowsM = []
    for r in thermal_rows:
        dT = full_cfg['T_allow'] - r['T_inner_end']
        ratio = full_cfg['T_allow'] / max(r['T_inner_end'], 1e-9)
        rowsM.append(dict(case=r['case'], T_peak=r['T_inner_end'], dT_margin=dT, ratio=ratio))
    df_margins = pd.DataFrame(rowsM)

    # Structural summaries (already have Pc_used stored)
    rowsS = []
    for r in thermal_rows:
        Pc_used = r['Pc_used']
        FoS, sig_hoop, sig_y = yield_FoS_at_T(Pc_used, full_cfg['D_core'], full_cfg['t_wall'], r['T_inner_end'])
        sig_allow = STRUCT['weld_eff'] * (sig_y / max(STRUCT['sf_allow'], 1e-9))
        Pc_allow = MAWP_from_yield(sig_allow, full_cfg['D_core'], full_cfg['t_wall'])
        rowsS.append(dict(case=r['case'], FoS=FoS, sigma_hoop_MPa=sig_hoop/1e6, sigma_y_MPa=sig_y/1e6,
                          MAWP_MPa=Pc_allow/1e6, Pc_used_MPa=Pc_used/1e6))
    df_struct = pd.DataFrame(rowsS)

    # Save CSVs
    df_flows.to_csv(full_cfg['outdir']/ "flows_phi.csv", index=False)
    df_thermal.to_csv(full_cfg['outdir']/"thermal_summary.csv", index=False)
    df_margins.to_csv(full_cfg['outdir']/"margins_summary.csv", index=False)
    df_energy.to_csv(full_cfg['outdir']/"energy_sanity.csv", index=False)
    df_ign.to_csv(full_cfg['outdir']/ "ignition_summary.csv", index=False)
    df_struct.to_csv(full_cfg['outdir']/"structural_summary.csv", index=False)

    # ---------- PLOTTING ----------
    # Slide 1 — Mass flows (labels already include pressures & T)
    plt.figure(figsize=(9,5))
    x = np.arange(len(df_flows))
    plt.bar(x-0.2, df_flows["mdot_O2_g_s"], width=0.4, label="O2")
    plt.bar(x+0.2, df_flows["mdot_H2_g_s"], width=0.4, label="H2")
    plt.xticks(x, df_flows["case"], rotation=12)
    plt.ylabel("Mass flow [g/s]")
    plt.title("Slide 1 — Mass flows by case")
    plt.grid(axis="y", linewidth=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(full_cfg['outdir']/"slide1_flows_phi.png", dpi=200); plt.close()

    # Slide 2 — Thermal per case (title reflects full case name & ramp)
    for i, r in enumerate(thermal_rows):
        # Re-run one series for plotting with same params
        t, Ti, To = simulate_two_node_ramped(
            r['h_hot_base'], r['h_H2'], A_inner, A_outer, A_cond, t_wall,
            r['T_gas_film'], full_cfg['T_H2_in'], full_cfg['rho_s'], full_cfg['cp_s'], full_cfg['k_s'],
            t_end=full_cfg['t_end'], dt=1e-3,
            t_ramp=full_cfg['t_ramp_hot'], ramp_shape=full_cfg['ramp_shape'],
            eps_rad=full_cfg['eps_rad'], T_rad_inf=full_cfg['T_rad_inf']
        )
        plt.figure(figsize=(9,5))
        plt.plot(t, Ti, linewidth=2, label="Inner wall")
        plt.plot(t, To, linewidth=2, linestyle="--", label="Outer wall")
        plt.axhline(full_cfg['T_allow'], linestyle=":", linewidth=1.5, label=f"Allowable = {full_cfg['T_allow']:.0f} K")
        plt.xlabel("Time [s]"); plt.ylabel("Temperature [K]")
        plt.title(f"Slide 2 — Thermal: {r['case']} (ramp={full_cfg['t_ramp_hot']} s, eps={full_cfg['eps_rad']})")
        plt.grid(True, linewidth=0.3); plt.legend(); plt.tight_layout()
        fname = f"slide2_thermal_case{i+1}.png"
        plt.savefig(full_cfg['outdir']/fname, dpi=200); plt.close()

    # Slide 3 — Thermal headroom
    plt.figure(figsize=(9,5))
    plt.bar(df_margins["case"], df_margins["dT_margin"])
    plt.axhline(100.0, linestyle="--", linewidth=1.5, label="Target 100 K")
    plt.ylabel("ΔT margin [K]")
    plt.title("Slide 3 — Thermal headroom (lower is worse)")
    plt.grid(axis="y", linewidth=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(full_cfg['outdir']/"slide3_thermal_margin.png", dpi=200); plt.close()

    # Slide 4 — Energy sanity
    plt.figure(figsize=(9,5))
    x = np.arange(len(df_energy))
    plt.bar(x-0.2, df_energy["Qsurf_W"]/1e3, width=0.4, label="Surface uptake")
    plt.bar(x+0.2, df_energy["Qavail_W"]/1e3, width=0.4, label="Avail chem (η=0.45)")
    plt.xticks(x, df_energy["case"], rotation=12)
    plt.ylabel("kW")
    plt.title("Slide 4 — Energy sanity")
    plt.grid(axis="y", linewidth=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(full_cfg['outdir']/"slide4_energy_sanity.png", dpi=200); plt.close()

    # Slide 5a — Yield FoS
    plt.figure(figsize=(9,5))
    plt.bar(df_struct["case"], df_struct["FoS"])
    plt.axhline(1.5, linestyle="--", linewidth=1.5, label="Target FoS = 1.5")
    plt.ylabel("FoS")
    plt.title("Slide 5 — Yield Factor of Safety")
    plt.grid(axis="y", linewidth=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(full_cfg['outdir']/"slide5_FoS.png", dpi=200); plt.close()

    # Slide 5b — MAWP vs Pc (bars vs dots)
    plt.figure(figsize=(9,5))
    plt.bar(df_struct["case"], df_struct["MAWP_MPa"])
    for i, row in df_struct.iterrows():
        plt.plot([i], [row['Pc_used_MPa']], marker='o')
    plt.ylabel("Pressure [MPa]")
    plt.title("Slide 5 — MAWP (bars) vs Pc (dots)")
    plt.grid(axis="y", linewidth=0.3); plt.tight_layout()
    plt.savefig(full_cfg['outdir']/"slide5_MAWP.png", dpi=200); plt.close()

    # Slide 6 — Ignition proxy (recirc & annulus velocity)
    plt.figure(figsize=(9,5))
    plt.bar(df_ign["case"], df_ign["U_ann"])
    plt.ylabel("Annulus U [m/s]")
    plt.title("Slide 6 — Ignition proxy (higher U may reduce Da)")
    plt.grid(axis="y", linewidth=0.3); plt.tight_layout()
    plt.savefig(full_cfg['outdir']/"slide6_ignition.png", dpi=200); plt.close()

    # Slide 7 — Uncertainty (Monte Carlo, first base case only for speed)
    if len(thermal_rows) > 0:
        rng = np.random.default_rng(11)
        r0 = thermal_rows[0]
        samples = []
        for _ in range(full_cfg['N_mc']):
            h_hot = r0['h_hot_base'] * rng.normal(1.0, full_cfg['sigma_h_hot'])
            h_cool= r0['h_H2']       * rng.normal(1.0, full_cfg['sigma_h_cool'])
            k_s   = full_cfg['k_s']  * rng.normal(1.0, full_cfg['sigma_k_s'])
            cp_s  = full_cfg['cp_s'] * rng.normal(1.0, full_cfg['sigma_cp_s'])
            Tgas  = r0['T_gas_film'] * rng.normal(1.0, full_cfg['sigma_Tgas'])
            _, Ti, _ = simulate_two_node_ramped(
                h_hot, h_cool, A_inner, A_outer, A_cond, t_wall,
                Tgas, full_cfg['T_H2_in'], full_cfg['rho_s'], cp_s, k_s,
                t_end=full_cfg['t_end'], dt=1e-3,
                t_ramp=full_cfg['t_ramp_hot'], ramp_shape=full_cfg['ramp_shape'],
                eps_rad=full_cfg['eps_rad'], T_rad_inf=full_cfg['T_rad_inf']
            )
            samples.append(Ti[-1])
        pd.DataFrame(dict(Tpeak_K=samples)).to_csv(full_cfg['outdir']/"uncertainty_samples_full.csv", index=False)
        p10, p50, p90 = np.percentile(samples, [10,50,90])

        plt.figure(figsize=(9,5))
        plt.hist(samples, bins=36)
        plt.axvline(full_cfg['T_allow'], linestyle="--", linewidth=1.5, label="Allowable")
        plt.xlabel("Peak inner-wall temperature [K]"); plt.ylabel("Count")
        plt.title("Slide 7 — Uncertainty (two-node Monte Carlo, ramped)")
        plt.grid(True, linewidth=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(full_cfg['outdir']/"slide7_uncertainty_band.png", dpi=200); plt.close()

    # Console summary
    summary = {
        "have_cantera": HAVE_CT,
        "Pc_model": full_cfg['Pc_model'],
        "t_end": full_cfg['t_end'],
        "t_ramp_hot": full_cfg['t_ramp_hot'],
        "film_factor": full_cfg['film_factor'],
        "eps_rad": full_cfg['eps_rad'],
        "flows_phi": df_flows.to_dict(orient="records"),
        "thermal_last": df_thermal.to_dict(orient="records"),
        "struct": df_struct.to_dict(orient="records"),
        "outdir": str(full_cfg['outdir'])
    }
    print(summary)


if __name__ == "__main__":
    run(cfg, SIMPLE_INPUTS)
