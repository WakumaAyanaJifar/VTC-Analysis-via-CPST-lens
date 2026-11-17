"""Step 3: held-out and uncertainty checks for validated-threshold coverage (VTC).

Usage:  python validate_vtc.py <detections.csv> <annotations.csv> <species.csv> <out_dir> [n_boot]

Three checks, all grouping by recording (the 33 one-hour files), because detections within a recording
are not independent:
  1. Grouped cross-validation: thresholds are fitted on 4/5 of the recordings and their precision is
     measured on the held-out 1/5 (5 folds). Reported per species and as a held-out VTC.
  2. Cluster bootstrap by recording: recordings are resampled with replacement and the whole procedure is
     repeated, giving 95% percentile intervals for VTC and for each species' threshold tau, and how often
     each species is validated.
  3. Lower-bound criterion: a species is validated only if the lower 95% confidence bound of the fitted
     probability of a true positive reaches p* (delta method on the logistic regression), which is
     stricter than requiring the point estimate to reach p*.
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm

det_csv, ann_csv, sp_csv, out_dir = sys.argv[1:5]
N_BOOT = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
os.makedirs(out_dir, exist_ok=True)
MIN_CONF, P_STAR, M_CLIPS, K_FOLDS, SEED = 0.10, 0.95, 50, 5, 20261117
Z = 1.959964
LOGIT_P = np.log(P_STAR / (1 - P_STAR))

det = pd.read_csv(det_csv)
ann = pd.read_csv(ann_csv).rename(columns={"Filename": "filename", "Start Time (s)": "a0", "End Time (s)": "a1",
                                           "Species eBird Code": "ebird_code"})
species = pd.read_csv(sp_csv)["Species eBird Code"].tolist()

ann_idx = {k: g[["a0", "a1"]].to_numpy() for k, g in ann.groupby(["filename", "ebird_code"])}
def is_tp(r):
    iv = ann_idx.get((r.filename, r.ebird_code))
    return bool(iv is not None and np.any((iv[:, 0] < r.end_s) & (iv[:, 1] > r.start_s)))
det["tp"] = np.array([is_tp(r) for r in det.itertuples()], dtype=float)
files = np.array(sorted(det.filename.unique()))
by_sp = {s: g[["filename", "confidence", "tp"]].reset_index(drop=True) for s, g in det.groupby("ebird_code")}


def fit(x, y, lower_bound=False):
    """Return tau (np.nan if the species cannot be validated). Same rules as compute_vtc.py:
    >= M_CLIPS detections, fitted P(TP) reaches P_STAR within the observed score range."""
    if len(y) < M_CLIPS or y.sum() == 0:
        return np.nan
    if y.sum() == len(y):
        return MIN_CONF
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sm.Logit(y, sm.add_constant(x)).fit(disp=0, maxiter=200)
    except Exception:
        return np.nan
    b0, b1 = res.params
    if b1 <= 0:
        return np.nan
    if not lower_bound:
        tau = (LOGIT_P - b0) / b1
        return max(tau, MIN_CONF) if tau <= x.max() else np.nan
    cov = res.cov_params()
    grid = np.linspace(MIN_CONF, x.max(), 2001)
    eta = b0 + b1 * grid
    se = np.sqrt(np.maximum(cov[0, 0] + grid ** 2 * cov[1, 1] + 2 * grid * cov[0, 1], 0))
    ok = np.nonzero(eta - Z * se >= LOGIT_P)[0]
    return grid[ok[0]] if len(ok) else np.nan


def vtc_for(frames, lower_bound=False):
    """frames: dict species -> DataFrame. Returns (taus dict, VTC, detection-weighted VTC, n_used)."""
    taus, used, n_det, n_val_det = {}, 0, 0, 0
    for s, g in frames.items():
        if len(g) == 0:
            continue
        used += 1; n_det += len(g)
        tau = fit(g.confidence.to_numpy(), g.tp.to_numpy(), lower_bound)
        taus[s] = tau
        if not np.isnan(tau):
            n_val_det += len(g)
    n_val = sum(not np.isnan(t) for t in taus.values())
    return taus, n_val / used, n_val_det / n_det, used


# ---- full-data reference (should match compute_vtc.py) -------------------------------------------
taus_full, vtc_full, vtcw_full, n_used = vtc_for(by_sp)
taus_lb, vtc_lb, vtcw_lb, _ = vtc_for(by_sp, lower_bound=True)
validated_full = sorted(s for s, t in taus_full.items() if not np.isnan(t))
validated_lb = sorted(s for s, t in taus_lb.items() if not np.isnan(t))

# ---- 1. grouped 5-fold cross-validation by recording ----------------------------------------------
rng = np.random.default_rng(SEED)
perm = rng.permutation(files)
folds = [set(perm[i::K_FOLDS]) for i in range(K_FOLDS)]
rows = []
for s in validated_full:
    g = by_sp[s]
    kept_tp = kept_n = 0; n_folds_fit = 0
    for test in folds:
        tr, te = g[~g.filename.isin(test)], g[g.filename.isin(test)]
        tau = fit(tr.confidence.to_numpy(), tr.tp.to_numpy())
        if np.isnan(tau):
            continue
        n_folds_fit += 1
        k = te[te.confidence >= tau]
        kept_n += len(k); kept_tp += k.tp.sum()
    prec = kept_tp / kept_n if kept_n else np.nan
    # Wilson 95% interval for held-out precision
    if kept_n:
        p, n = prec, kept_n
        c = (p + Z**2 / (2*n)) / (1 + Z**2 / n); h = Z * np.sqrt(p*(1-p)/n + Z**2/(4*n*n)) / (1 + Z**2 / n)
        lo, hi = c - h, c + h
    else:
        lo = hi = np.nan
    rows.append(dict(ebird_code=s, tau_full=round(taus_full[s], 3), folds_with_threshold=n_folds_fit,
                     heldout_kept=int(kept_n), heldout_precision=round(prec, 3) if kept_n else np.nan,
                     heldout_ci_low=round(lo, 3) if kept_n else np.nan, heldout_ci_high=round(hi, 3) if kept_n else np.nan,
                     heldout_meets_pstar=bool(kept_n and prec >= P_STAR),
                     heldout_status=("no held-out detections above tau" if kept_n == 0 else
                                     "meets p*" if prec >= P_STAR else "below p*"),
                     validated_lower_bound=s in validated_lb))
cv = pd.DataFrame(rows)
n_hold = int(cv.heldout_meets_pstar.sum())
vtc_cv = n_hold / n_used

# ---- 2. cluster bootstrap by recording --------------------------------------------------------------
boot_vtc, boot_vtcw = [], []
val_count = {s: 0 for s in by_sp}
tau_draws = {s: [] for s in validated_full}
groups = {s: {f: g[g.filename == f] for f in g.filename.unique()} for s, g in by_sp.items()}
for b in range(N_BOOT):
    draw = rng.choice(files, size=len(files), replace=True)
    frames = {}
    for s, gs in groups.items():
        parts = [gs[f] for f in draw if f in gs]
        frames[s] = pd.concat(parts, ignore_index=True) if parts else by_sp[s].iloc[0:0]
    taus, v, vw, _ = vtc_for(frames)
    boot_vtc.append(v); boot_vtcw.append(vw)
    for s, t in taus.items():
        if not np.isnan(t):
            val_count[s] += 1
            if s in tau_draws:
                tau_draws[s].append(t)
    if (b + 1) % 100 == 0:
        print(f"bootstrap {b + 1}/{N_BOOT}", flush=True)

def pct(a, q): return round(float(np.percentile(a, q)), 3)
cv["validated_in_bootstrap_share"] = [round(val_count[s] / N_BOOT, 3) for s in cv.ebird_code]
cv["tau_ci_low"] = [pct(tau_draws[s], 2.5) if tau_draws[s] else np.nan for s in cv.ebird_code]
cv["tau_ci_high"] = [pct(tau_draws[s], 97.5) if tau_draws[s] else np.nan for s in cv.ebird_code]
others = sorted((s for s in by_sp if s not in validated_full and val_count[s] > 0), key=lambda s: -val_count[s])

summary = dict(
    settings=dict(p_star=P_STAR, m=M_CLIPS, folds=K_FOLDS, n_boot=N_BOOT, seed=SEED, n_recordings=len(files)),
    full_data=dict(n_used=n_used, n_validated=len(validated_full), VTC=round(vtc_full, 3), VTC_detection_weighted=round(vtcw_full, 3)),
    cross_validation=dict(n_validated_full=len(validated_full), n_meeting_pstar_heldout=n_hold, VTC_heldout=round(vtc_cv, 3),
                          species_below_pstar_heldout=cv.loc[cv.heldout_status == "below p*", "ebird_code"].tolist(),
                          species_no_heldout_detections=cv.loc[cv.heldout_status.str.startswith("no"), "ebird_code"].tolist(),
                          median_heldout_precision=round(float(cv.heldout_precision.median()), 3)),
    bootstrap=dict(VTC_median=pct(boot_vtc, 50), VTC_95ci=[pct(boot_vtc, 2.5), pct(boot_vtc, 97.5)],
                   VTC_weighted_95ci=[pct(boot_vtcw, 2.5), pct(boot_vtcw, 97.5)],
                   n_validated_species_stable_ge_0_9=int(sum(val_count[s] / N_BOOT >= 0.9 for s in validated_full)),
                   other_species_sometimes_validated={s: round(val_count[s] / N_BOOT, 3) for s in others}),
    lower_bound_criterion=dict(n_validated=len(validated_lb), VTC=round(vtc_lb, 3), VTC_detection_weighted=round(vtcw_lb, 3),
                               dropped_vs_point_estimate=sorted(set(validated_full) - set(validated_lb))),
)
cv.to_csv(os.path.join(out_dir, "vtc_heldout_species.csv"), index=False)
json.dump(summary, open(os.path.join(out_dir, "vtc_heldout_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
print(cv.to_string(index=False))
