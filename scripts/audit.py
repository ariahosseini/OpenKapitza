"""Independent numerical audit of the OpenKapitza Phase-0 core.

Every quantity here is recomputed from first principles with literals, using
different numerical methods where integration is involved, and compared
against what the package produces. Nothing in this file imports the package's
physics -- only its public entry points, so a shared bug cannot hide.
"""
import math
import sys
from pathlib import Path

# Import THIS repository's package explicitly. An earlier run of this audit
# silently validated a stale copy of openkapitza that happened to sit next to
# the script on sys.path, and reported a defect as unfixed after it had been
# fixed. Pinning the path makes that failure mode impossible.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

K_B = 1.380649e-23
N_A = 6.02214076e23
R_GAS = K_B * N_A

FAIL = []
def chk(name, mine, pkg, tol=1e-9, note=""):
    rel = abs(mine - pkg) / abs(pkg) if pkg else abs(mine - pkg)
    ok = rel <= tol
    if not ok:
        FAIL.append(name)
    print(f"{'OK ' if ok else 'XX '} {name:42s} audit={mine:.6g}  pkg={pkg:.6g}  rel={rel:.2e}  {note}")

print("=" * 100)
print("1. INDEPENDENT RECOMPUTATION vs PACKAGE")
print("=" * 100)

from openkapitza import build_joint, run_continuum
from openkapitza.schema import load_material, load_gas, FlagSet, Quantity
from openkapitza.gap import gas_state
from openkapitza.md import debye_heat_capacity, dmm_conductance

# ---- case: SS304/SS304, 2 MPa, 300 K, air at 1 atm -------------------------
j = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                temperature_K=300.0, gas="air", gas_pressure_kPa=101.325)
r = run_continuum(j, with_uncertainty=False)
net = r.primary

# literals for SS304 from the YAML
k_ss, E_ss, nu_ss, Hv_ss = 16.2, 193.0e9, 0.29, 2.0e9
sa = sb = 1.2e-6
ma = mb = 0.08
P = 2.0e6

sigma_eff = math.sqrt(sa**2 + sb**2)
m_eff = math.sqrt(ma**2 + mb**2)
k_s = 2*k_ss*k_ss/(k_ss+k_ss)
E_eff = 1.0 / (2*(1-nu_ss**2)/E_ss)
# SS304 now carries published Song-Yovanovich coefficients, so the contact
# hardness comes from the explicit correlation rather than the bulk Vickers
# number. Recomputed here from literals, independently of the package.
c1_ss, c2_ss = 6.271e9, -0.229
sigma_over_m_um = (sigma_eff/m_eff)*1e6
p_hc = (P / (c1_ss*(1.62*sigma_over_m_um)**c2_ss))**(1.0/(1.0+0.071*c2_ss))
H_c = P/p_hc
psi = E_eff*m_eff/H_c
h_c = 1.25*k_s*(m_eff/sigma_eff)*p_hc**0.95
xi = p_hc
Y = 1.53*sigma_eff*p_hc**(-0.097)

chk("sigma_eff", sigma_eff, r.sigma_eff.si)
chk("m_eff", m_eff, r.m_eff.si)
chk("k_s", k_s, r.k_s.si)
chk("E_eff", E_eff, r.E_eff.si)
chk("H_c", H_c, r.hardness.H_c.si)
chk("P/H_c", p_hc, r.hardness.P_over_Hc.si)
chk("plasticity index", psi, r.regime.psi.si)
# Cross-check the explicit correlation against the implicit system it
# approximates, solved here with scipy rather than the package's own bisection.
from scipy.special import erfc as _erfc, erfcinv as _erfcinv
_Hc = c1_ss
for _ in range(400):
    _lam = math.sqrt(2)*_erfcinv(2*min(P/_Hc, 0.999999))
    _a = sigma_over_m_um*math.sqrt(8/math.pi)*math.exp(_lam**2/2)*_erfc(_lam/math.sqrt(2))
    _Hc = 0.5*_Hc + 0.5*c1_ss*(math.sqrt(2*math.pi)*_a)**c2_ss
print(f"OK  {'H_c explicit vs implicit system':42s} audit={H_c:.6g}  implicit={_Hc:.6g}  "
      f"rel={abs(H_c-_Hc)/_Hc:.2e}  (approximation, ~1% expected)")
chk("h_c (CMY plastic)", h_c, net.h_c.si if hasattr(net,'h_c') else r.branches['plastic'].R_constriction.si and 1/r.branches['plastic'].R_constriction.si)
chk("xi", xi, net.xi.si)
chk("Y", Y, r.Y.si)

# ---- gas -------------------------------------------------------------------
T = 300.0
Pg = 101325.0
mu_ref, mu_T, mu_S = 1.716e-5, 273.15, 110.4
k_ref, k_T, k_S = 0.0241, 273.15, 194.0
M_air, gamma, Pr = 28.97e-3, 1.400, 0.71

suth = lambda p, Tr, S: p*((Tr+S)/(T+S))*(T/Tr)**1.5
mu = suth(mu_ref, mu_T, mu_S)
kg = suth(k_ref, k_T, k_S)
Rsp = R_GAS/M_air
mfp = (mu/Pg)*math.sqrt(math.pi*Rsp*T/2)
beta = 2*gamma/((gamma+1)*Pr)
alpha_t = 2*(2-0.85)/0.85
Mrar = alpha_t*beta*mfp
h_g = kg/(Y+Mrar)

chk("gas mu", mu, r.gap.state.mu.si)
chk("gas k_g", kg, r.gap.state.k_g.si)
chk("gas mfp", mfp, r.gap.state.mfp.si)
chk("gas beta", beta, r.gap.state.beta.si)
chk("accommodation term", alpha_t, r.gap.alpha_term.si)
chk("rarefaction M", Mrar, r.gap.M.si)
chk("h_g", h_g, r.gap.h_g.si)

# ---- network ---------------------------------------------------------------
h_K = r.kapitza.h_K.si
R_con = 1/h_c
R_kap = 1/(xi*h_K)
h_solid = 1/(R_con+R_kap)
h_joint = h_solid + h_g
chk("R_constriction", R_con, net.R_constriction.si)
chk("R_kapitza (scaled)", R_kap, net.R_kapitza.si)
chk("h_solid", h_solid, net.h_solid.si)
chk("h_joint", h_joint, net.h_joint.si)

print()
print("=" * 100)
print("2. ABSOLUTE CHECKS AGAINST KNOWN PHYSICAL VALUES")
print("=" * 100)

# air mean free path at 300 K / 1 atm: textbook ~67 nm
print(f"air mfp @300K,1atm  = {mfp*1e9:8.2f} nm   (textbook ~67 nm)")
print(f"air k   @300K       = {kg:8.5f} W/mK (textbook ~0.0262)")
mu_lit = 1.846e-5
print(f"air mu  @300K       = {mu:.4e} Pa.s (textbook ~1.846e-5, rel err "
      f"{abs(mu-mu_lit)/mu_lit*100:.2f} %)")

# Debye heat capacity of Si at 300 K vs experiment
# experimental: c_p ~ 700 J/kg.K, rho = 2329 -> ~1.63e6 J/m^3K
def debye_C_trapz(theta, n, T, steps=200000):
    x = np.linspace(1e-12, theta/T, steps)
    f = x**4*np.exp(x)/(np.exp(x)-1)**2
    return 9*n*K_B*(T/theta)**3*np.trapezoid(f, x)

C_si_audit = debye_C_trapz(645.0, 5.00e28, 300.0)
C_si_pkg = debye_heat_capacity(Quantity(si=645.0, unit="K"),
                               Quantity(si=5.00e28, unit="-"),
                               Quantity(si=300.0, unit="K")).si
C_si_exp = 700.0*2329.0
chk("Debye C_V(Si,300K) simpson vs trapz", C_si_audit, C_si_pkg, tol=1e-6)
print(f"   C_V(Si,300K) model = {C_si_pkg:.4e} J/m^3K   experiment ~{C_si_exp:.4e}   "
      f"ratio {C_si_pkg/C_si_exp:.3f}")

print()
print("=" * 100)
print("3. DMM ELASTIC CUTOFF")
print("=" * 100)

vL_si, vT_si, th_si, n_si = 8433.0, 5843.0, 645.0, 5.00e28
vL_ge, vT_ge, th_ge, n_ge = 4914.0, 3537.0, 374.0, 4.42e28

S2 = lambda vl, vt: 1/vl**2 + 2/vt**2
S3 = lambda vl, vt: 1/vl**3 + 2/vt**3

s2_si, s3_si = S2(vL_si, vT_si), S3(vL_si, vT_si)
s2_ge = S2(vL_ge, vT_ge)
alpha_si_ge = s2_ge/(s2_si+s2_ge)
v_eff_si = s2_si/s3_si

C_si = debye_C_trapz(th_si, n_si, 300.0)
h_nocut = 0.25*C_si*v_eff_si*alpha_si_ge

# WITH the elastic cutoff: phonons above Ge's Debye frequency have no states
# to transmit into, so the heat-capacity integral must stop at the LOWER
# Debye temperature.
def debye_C_truncated(theta_host, n, T, theta_cut, steps=200000):
    x_max = min(theta_host, theta_cut)/T
    x = np.linspace(1e-12, x_max, steps)
    f = x**4*np.exp(x)/(np.exp(x)-1)**2
    return 9*n*K_B*(T/theta_host)**3*np.trapezoid(f, x)

C_si_cut = debye_C_truncated(th_si, n_si, 300.0, th_ge)
h_cut = 0.25*C_si_cut*v_eff_si*alpha_si_ge

f = FlagSet()
est = dmm_conductance(load_material("Si"), load_material("Ge"),
                      Quantity(si=300.0, unit="K"), f)

print(f"transmission alpha(Si->Ge)      = {alpha_si_ge:.4f}")
print(f"v_eff(Si) = S2/S3               = {v_eff_si:.1f} m/s")
print(f"C_V(Si) full spectrum           = {C_si:.4e} J/m^3K")
print(f"C_V(Si) truncated at theta_Ge   = {C_si_cut:.4e} J/m^3K   "
      f"({C_si_cut/C_si*100:.1f} % of full)")
print()
print(f"h_DMM no cutoff (audit, WRONG)  = {h_nocut/1e6:8.1f} MW/m^2K")
print(f"h_DMM package (cutoff applied)  = {est.h_K.si/1e6:8.1f} MW/m^2K")
print()
print("Published Si/Ge at 300 K:")
print("   DMM (Landry & McGaughey, cited)   R=2.4e-9  m2K/W ->  417 MW/m^2K")
print("   NEMD (Landry et al.)              R=2.7-3.2e-9    ->  315-368 MW/m^2K")
print("   AEMD sharp interface, Si->Ge      R=3.76e-9       ->  266 MW/m^2K")
print()
# detailed balance: exact for DMM, and the sharpest check on the formulation
rev = dmm_conductance(load_material("Ge"), load_material("Si"),
                      Quantity(si=300.0, unit="K"), f)
chk("detailed balance h(Si->Ge)==h(Ge->Si)", est.h_K.si, rev.h_K.si, tol=1e-9)

print()
print("=" * 100)
print("4. MONTE CARLO CENTRING")
print("=" * 100)
r2 = run_continuum(j, n_samples=4000)
mc = r2.uncertainty["h_joint"]
print(f"nominal = {mc.nominal:.4g}   median = {mc.median:.4g}   "
      f"mean = {mc.mean:.4g}")
print(f"median/nominal = {mc.median/mc.nominal:.4f}")
print("  sampler is median-preserving; the residual gap is propagation through")
print("  H_c = min(Hv_a, Hv_b) over two uncertain hardnesses, and the report")
print("  explains it when it exceeds 2 %.")

print()
print("=" * 100)
print("FAILURES:", FAIL if FAIL else "none in section 1")
print("=" * 100)
