"""
===============================================================================
ex8 - Near photon-sphere ray-tracer validation
===============================================================================
Stress-tests the integrator in its most demanding regime: equatorial photons
with impact parameter b approaching the critical value
    b_crit = 3*sqrt(3) * M   (Schwarzschild photon sphere)
Two diagnostics:

  TEST 1 - Bisection for b_crit
           How many significant digits of b_crit does the code resolve before
           escape/capture classification becomes noise?

  TEST 2 - Orbit count & H-drift vs proximity to b_crit
           For b = b_crit*(1 +/- 10^-k), k = 1..12, record:
             - number of photon-sphere loops (phi advance / 2*pi)
             - maximum Hamiltonian drift |H_max - H_0|
             - minimum radius reached
           The loop count should diverge as ~ -ln(eps)/(2*pi) (Darwin 1959).
           The H-drift should grow roughly linearly with loop count if the
           integrator is honest (pure accumulation, no instability).

Uses Schwarzschild (a=0 Kerr), so all analytical benchmarks are exact.
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

from scr.black_holes import kerr
from scr.common.integrator import integrate, make_events

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
M = 1.0
b_crit_analytical = 3.0 * sqrt(3.0) * M        # 5.196152422706632...

bh = kerr.BlackHole(a=0.0)                     # Schwarzschild
r_horizon = bh.EH                              # = 2 M
r0        = 1000.0                             # launch radius (far, ~flat)
r_esc     = 2.0 * r0                           # "escaped" threshold
lmbda_max = 5.0 * r0                           # budget per trajectory
rtol, atol = 1e-10, 1e-12


def initial_state(b):
    """Equatorial photon at (r=r0, theta=pi/2, phi=0), moving inward.
    Convention: E = -k_t = 1, L = k_phi = b.  k_r < 0 (infalling)."""
    r = r0
    k_t, k_th, k_ph = -1.0, 0.0, b
    one_minus = 1.0 - 2.0*M/r                  # (1 - 2M/r)
    # Null condition (Schwarzschild, equatorial):
    #   -k_t^2/one_minus + one_minus*k_r^2 + k_ph^2/r^2 = 0
    k_r_sq = (k_t*k_t/one_minus - k_ph*k_ph/(r*r)) / one_minus
    k_r = -sqrt(k_r_sq)
    return np.array([0.0, r, 0.5*pi, 0.0, k_t, k_r, k_th, k_ph])


def rhs(lmbda, q):
    return bh.geodesics(q, lmbda)


def hamiltonian(Y):
    """|H(lambda)| along trajectory, where H = 1/2 g^{mu nu} k_mu k_nu."""
    H = np.zeros(len(Y))
    for i, s in enumerate(Y):
        gtt, grr, gthth, gphph, gtph = bh.inverse_metric(s[:4])
        k = s[4:]
        H[i] = 0.5*(gtt*k[0]*k[0] + grr*k[1]*k[1]
                    + gthth*k[2]*k[2] + gphph*k[3]*k[3]
                    + 2*gtph*k[0]*k[3])
    return H


def run(b):
    """Integrate one photon and return (status, n_loops, dH, r_min)."""
    events = make_events(bh, acc_structure=None, r_escape=r_esc)
    y0 = initial_state(b)
    res = integrate(rhs, y0, (0.0, lmbda_max),
                    method="DOP853", events=events,
                    rtol=rtol, atol=atol)
    # n_loops from total phi traveled
    n_loops = abs(res.y[-1, 3] - res.y[0, 3]) / (2.0*pi)
    H = hamiltonian(res.y)
    dH = float(np.max(np.abs(H - H[0])))
    r_min = float(np.min(res.y[:, 1]))
    return res.status, n_loops, dH, r_min


# -----------------------------------------------------------------------------
# TEST 1 - Bisection
# -----------------------------------------------------------------------------
print("="*72)
print("TEST 1 - Critical impact parameter via bisection")
print("="*72)
print(f"Analytical  b_crit = 3*sqrt(3) M = {b_crit_analytical:.15f}\n")

b_lo, b_hi = 5.0, 6.0
assert run(b_lo)[0] == "horizon", "lower bracket must capture"
assert run(b_hi)[0] == "escape",  "upper bracket must escape"

for it in range(60):
    b_mid = 0.5*(b_lo + b_hi)
    status, *_ = run(b_mid)
    if status == "horizon":
        b_lo = b_mid
    else:
        b_hi = b_mid
    if b_hi - b_lo < 1e-15:
        break

b_crit_num = 0.5*(b_lo + b_hi)
abs_err = abs(b_crit_num - b_crit_analytical)
rel_err = abs_err / b_crit_analytical
digits  = -np.log10(max(rel_err, 1e-17))

print(f"Iterations       : {it+1}")
print(f"Numerical b_crit : {b_crit_num:.15f}")
print(f"Absolute error   : {abs_err:.3e}")
print(f"Relative error   : {rel_err:.3e}")
print(f"Resolved digits  : {digits:.1f}")
print(f"Expected ceiling : ~ -log10(rtol) = {-np.log10(rtol):.0f}")

if digits >= 8:
    verdict = "EXCELLENT - integrator resolves b_crit near its theoretical limit"
elif digits >= 5:
    verdict = "OK - resolution consistent with rtol"
else:
    verdict = "POOR - something is accumulating more error than expected"
print(f"Verdict          : {verdict}\n")


# -----------------------------------------------------------------------------
# TEST 2 - Orbits & H-drift vs eps = |b/b_crit - 1|
# -----------------------------------------------------------------------------
print("="*72)
print("TEST 2 - Orbit count and Hamiltonian drift as b -> b_crit")
print("="*72)

ks  = np.arange(1, 13)
eps = 10.0 ** (-ks.astype(float))

print("\n> b > b_crit (grazes then escapes)")
print(f"{'k':>3} {'eps':>10} {'status':>10} {'n_loops':>10} "
      f"{'|dH|':>12} {'r_min':>10}")
print("-"*60)
above = []
for k, e in zip(ks, eps):
    s, n, dH, rm = run(b_crit_analytical*(1.0 + e))
    above.append((n, dH, rm, s))
    print(f"{k:3d} {e:10.1e} {s:>10} {n:10.2f} {dH:12.3e} {rm:10.5f}")

print("\n> b < b_crit (grazes then plunges)")
print(f"{'k':>3} {'eps':>10} {'status':>10} {'n_loops':>10} "
      f"{'|dH|':>12} {'r_min':>10}")
print("-"*60)
below = []
for k, e in zip(ks, eps):
    s, n, dH, rm = run(b_crit_analytical*(1.0 - e))
    below.append((n, dH, rm, s))
    print(f"{k:3d} {e:10.1e} {s:>10} {n:10.2f} {dH:12.3e} {rm:10.5f}")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
n_a  = np.array([d[0] for d in above])
dH_a = np.array([d[1] for d in above])
n_b  = np.array([d[0] for d in below])
dH_b = np.array([d[1] for d in below])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogx(eps, n_a, 'o-', label=r'$b > b_\mathrm{crit}$ (escape)')
ax.semilogx(eps, n_b, 's-', label=r'$b < b_\mathrm{crit}$ (plunge)')
# Darwin 1959 asymptotic slope: n_loops ~ -ln(eps)/(2*pi) + const
ax.semilogx(eps, -np.log(eps)/(2*pi), 'k--', alpha=0.5,
            label=r'$-\ln\epsilon/(2\pi)$  (Darwin 1959)')
ax.set_xlabel(r'$\epsilon = |b/b_\mathrm{crit} - 1|$')
ax.set_ylabel('Number of photon-sphere loops')
ax.invert_xaxis()
ax.grid(alpha=0.3)
ax.legend()
ax.set_title(r'Logarithmic divergence of $n_\mathrm{loops}$')

ax = axes[1]
ax.loglog(n_a, dH_a, 'o-', label=r'$b > b_\mathrm{crit}$')
ax.loglog(n_b, dH_b, 's-', label=r'$b < b_\mathrm{crit}$')
# Reference line: linear accumulation dH ~ n_loops * rtol
ref_n = np.linspace(1, max(n_a.max(), n_b.max()), 50)
ax.loglog(ref_n, rtol*ref_n, 'k--', alpha=0.5,
          label=fr'$\propto n_\mathrm{{loops}} \cdot$ rtol')
ax.set_xlabel(r'Number of photon-sphere loops')
ax.set_ylabel(r'$|\Delta H|_\mathrm{max}$')
ax.grid(alpha=0.3, which='both')
ax.legend()
ax.set_title('Error accumulation: H-drift vs orbit count')

plt.tight_layout()
plt.savefig('images/ex8_near_photon_sphere.png', dpi=200, bbox_inches='tight')
print("\nPlot saved to images/ex8_near_photon_sphere.png")
plt.show()


# =============================================================================
# TEST 3 - Near-horizon behavior (radial infall, b = 0)
# =============================================================================
# For pure radial null geodesics in Schwarzschild, there is a closed form:
#     dr/dlambda = -E   =>   r(lambda) = r0 - E*lambda
# while k_r(r) = -E/(1-2M/r) diverges at the horizon (coordinate singularity).
# We scan the horizon margin eps_horizon and the integrator tolerance rtol
# to measure exactly how much each one contributes to the ~1e-7 H-drift
# seen in Test 2.
# =============================================================================
print("\n" + "="*72)
print("TEST 3 - Near-horizon behavior (radial infall)")
print("="*72)


def initial_state_radial():
    """Pure radial null geodesic, E=1, L=0, ingoing."""
    one_minus = 1.0 - 2.0*M/r0
    return np.array([0.0, r0, 0.5*pi, 0.0,
                     -1.0, -1.0/one_minus, 0.0, 0.0])


def run_radial(eps_h=1e-3, rtol_local=rtol, atol_local=atol):
    events = make_events(bh, acc_structure=None, r_escape=r_esc,
                         eps_horizon=eps_h)
    y0 = initial_state_radial()
    res = integrate(rhs, y0, (0.0, lmbda_max),
                    method="DOP853", events=events,
                    rtol=rtol_local, atol=atol_local)
    H = hamiltonian(res.y)
    dH = float(np.max(np.abs(H - H[0])))
    # Analytical check: r(lambda) = r0 - lambda exactly
    r_ana = r0 - res.t
    dev = float(np.max(np.abs(res.y[:, 1] - r_ana)))
    return {
        'status': res.status,
        'dH': dH,
        'r_min': float(np.min(res.y[:, 1])),
        'nfev': int(res.nfev),
        'dev_radial': dev,
        't': res.t,
        'y': res.y,
    }


# --- Part A: vary eps_horizon (with rtol fixed at 1e-10) --------------------
print("\n> Part A - Scanning eps_horizon at rtol = 1e-10")
print(f"{'eps_h':>10} {'status':>10} {'r_min':>10} {'nfev':>8} "
      f"{'|dH|_max':>12} {'|r - r_ana|':>14}")
print("-"*72)
eps_h_values = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
partA = []
for eh in eps_h_values:
    d = run_radial(eps_h=eh)
    partA.append((eh, d))
    print(f"{eh:10.1e} {d['status']:>10} {d['r_min']:10.5f} {d['nfev']:8d} "
          f"{d['dH']:12.3e} {d['dev_radial']:14.3e}")

# --- Part B: vary rtol at fixed eps_horizon = 1e-3 --------------------------
print("\n> Part B - Scanning rtol at eps_horizon = 1e-3")
print(f"{'rtol':>10} {'status':>10} {'nfev':>8} {'|dH|_max':>12} "
      f"{'|r - r_ana|':>14}")
print("-"*72)
rtol_values = [1e-6, 1e-8, 1e-10, 1e-12]
partB = []
for rt in rtol_values:
    d = run_radial(eps_h=1e-3, rtol_local=rt, atol_local=rt*1e-2)
    partB.append((rt, d))
    print(f"{rt:10.1e} {d['status']:>10} {d['nfev']:8d} "
          f"{d['dH']:12.3e} {d['dev_radial']:14.3e}")

# --- Plots -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (i) |dH| and deviation vs eps_horizon
ax = axes[0]
ehs      = np.array([p[0]         for p in partA])
dH_vs_eh = np.array([p[1]['dH']   for p in partA])
dev_vs_eh = np.array([p[1]['dev_radial'] for p in partA])
nfev_vs_eh = np.array([p[1]['nfev'] for p in partA])
ax.loglog(ehs, dH_vs_eh,  'o-', label=r'$|\Delta H|_\mathrm{max}$')
ax.loglog(ehs, dev_vs_eh, 's-', label=r'$\max |r_\mathrm{num} - r_\mathrm{ana}|$')
ax.set_xlabel(r'horizon margin $\epsilon_h$  (stop at $r = 2M + \epsilon_h$)')
ax.set_ylabel('error')
ax.invert_xaxis()
ax.grid(alpha=0.3, which='both')
ax.legend()
ax.set_title('Error vs horizon margin')

# (ii) nfev vs eps_horizon (integrator work)
ax = axes[1]
ax.loglog(ehs, nfev_vs_eh, 'd-', color='tab:purple')
ax.set_xlabel(r'horizon margin $\epsilon_h$')
ax.set_ylabel('integrator steps (nfev)')
ax.invert_xaxis()
ax.grid(alpha=0.3, which='both')
ax.set_title('Integrator work grows as step size shrinks near horizon')

# (iii) k_r(r) for one trajectory, showing the coordinate divergence
ax = axes[2]
d_show = run_radial(eps_h=1e-4)
r_traj = d_show['y'][:, 1]
kr_traj = d_show['y'][:, 5]
r_fine  = np.linspace(2.0 + 1e-4, r0, 2000)
kr_ana  = -1.0 / (1.0 - 2.0/r_fine)
ax.plot(r_fine, -kr_ana, 'k-', alpha=0.6, label=r'$-k_r^\mathrm{ana}(r)$')
ax.plot(r_traj, -kr_traj, 'o', ms=3, label='numerical accepted steps')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('r / M')
ax.set_ylabel(r'$-k_r$  (log)')
ax.set_title(r'Coordinate singularity: $k_r \propto 1/(1-2M/r)$')
ax.grid(alpha=0.3, which='both')
ax.legend()

plt.tight_layout()
plt.savefig('images/ex8_near_horizon.png', dpi=200, bbox_inches='tight')
print("\nPlot saved to images/ex8_near_horizon.png")
plt.show()

# --- Part C: horizon-event detection precision -----------------------------
# For radial infall we have an exact analytical solution:
#     r(lambda) = r0 - lambda      (since dr/dlambda = -E = -1)
# so the horizon event r = r_EH + eps_h should fire at
#     lambda_ana = r0 - r_EH - eps_h
# We compare this exact value against the numerical lambda returned by the
# event-refinement machinery (scipy dense output for DOP853, Brent-on-Hermite
# for the in-house RK45). Also measure how tightly r at the event matches
# the target cutoff r_EH + eps_h.
# ---------------------------------------------------------------------------
def run_radial_method(eps_h, method):
    events = make_events(bh, acc_structure=None, r_escape=r_esc,
                         eps_horizon=eps_h)
    y0 = initial_state_radial()
    res = integrate(rhs, y0, (0.0, lmbda_max),
                    method=method, events=events,
                    rtol=rtol, atol=atol)
    lam_num   = float(res.t[-1])
    r_event   = float(res.y[-1, 1])
    lam_ana   = r0 - r_horizon - eps_h
    r_target  = r_horizon + eps_h
    return {
        'status': res.status,
        'lam_num': lam_num,
        'lam_ana': lam_ana,
        'dlambda': abs(lam_num - lam_ana),
        'r_event': r_event,
        'dr_event': abs(r_event - r_target),
        'nfev': int(res.nfev),
    }


print("\n> Part C - Horizon-event detection precision (radial infall)")
print(f"  Exact analytical lambda at event : lambda_ana = r0 - r_EH - eps_h")
print(f"  Brent xtol for event refinement  : 1e-12 (in-house RK45)\n")

eps_h_scan = [1e-2, 1e-3, 1e-4, 1e-5]
print(f"{'method':>8} {'eps_h':>8} {'lambda_num':>18} {'lambda_ana':>18} "
      f"{'|dlambda|':>12} {'|dr_ev|':>12} {'nfev':>6}")
print("-"*88)
partC = []
for method in ("DOP853", "RK45"):
    for eh in eps_h_scan:
        d = run_radial_method(eh, method)
        partC.append((method, eh, d))
        print(f"{method:>8} {eh:8.1e} {d['lam_num']:18.12f} "
              f"{d['lam_ana']:18.12f} {d['dlambda']:12.3e} "
              f"{d['dr_event']:12.3e} {d['nfev']:6d}")


# --- Diagnostic summary ----------------------------------------------------
print("\n" + "="*72)
print("NEAR-HORIZON DIAGNOSTIC SUMMARY")
print("="*72)
slope_eh  = np.polyfit(np.log10(ehs), np.log10(dH_vs_eh), 1)[0]
rtols_B   = np.array([p[0]           for p in partB])
dHs_B     = np.array([p[1]['dH']     for p in partB])
slope_rt  = np.polyfit(np.log10(rtols_B), np.log10(dHs_B), 1)[0]
print(f"|dH| vs eps_horizon : d log|dH| / d log(eps_h) = {slope_eh:+.2f}")
print(f"  (<0 means tighter cutoff hurts, >0 means more margin helps)")
print(f"|dH| vs rtol        : d log|dH| / d log(rtol)  = {slope_rt:+.2f}")
print(f"  (slope ~ 1 means error is linearly controlled by rtol)")
print("="*72)
