"""Step 2: compute validated-threshold coverage (VTC) from BirdNET detections and expert annotations.
Usage: python compute_vtc.py <detections.csv> <annotations.csv> <species.csv> <out_dir>
A detection (species, 3-s window) is a true positive if an annotation box of that species in the same
file overlaps the window. For each species, a logistic regression of true/false-positive status on the
confidence score gives the threshold tau at which P(true positive) >= p_star (Wood & Kahl 2024; Symes et al. 2024)."""
import sys, os, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm

det_csv, ann_csv, sp_csv, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
MIN_CONF = 0.10          # floor used when running BirdNET
P_STAR, M_CLIPS = 0.95, 50   # headline settings; sensitivity below

det = pd.read_csv(det_csv)
ann = pd.read_csv(ann_csv).rename(columns={"Filename": "filename", "Start Time (s)": "a0", "End Time (s)": "a1",
                                           "Species eBird Code": "ebird_code"})
sp = pd.read_csv(sp_csv)

# --- label detections as true/false positive -------------------------------------------------
ann_idx = {k: g[["a0", "a1"]].to_numpy() for k, g in ann.groupby(["filename", "ebird_code"])}
def is_tp(r):
    iv = ann_idx.get((r.filename, r.ebird_code))
    return bool(iv is not None and np.any((iv[:, 0] < r.end_s) & (iv[:, 1] > r.start_s)))
det["tp"] = [is_tp(r) for r in det.itertuples()]

# --- annotated 3-s windows per species (for recall) -------------------------------------------
ann_windows = {}
for (fn, code), iv in ann_idx.items():
    w = set()
    for a0, a1 in iv:
        w.update(range(int(a0 // 3), int(np.ceil(a1 / 3))))
    ann_windows.setdefault(code, set()).update((fn, k) for k in w)
det["win"] = (det.start_s // 3).astype(int)

def fit_threshold(g, p_star):
    """Return (status, tau, b0, b1). tau = lowest confidence with fitted P(TP) >= p_star."""
    y = g.tp.to_numpy().astype(float); x = g.confidence.to_numpy()
    if y.sum() == 0: return "no true positives", np.nan, np.nan, np.nan
    if y.sum() == len(y): return "ok (no false positives)", MIN_CONF, np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sm.Logit(y, sm.add_constant(x)).fit(disp=0, maxiter=200)
        b0, b1 = res.params
    except Exception as e:
        return "fit failed (separation)", np.nan, np.nan, np.nan
    if b1 <= 0: return "score not informative (slope <= 0)", np.nan, b0, b1
    tau = (np.log(p_star / (1 - p_star)) - b0) / b1
    if tau > 1.0: return "p* not reached at any score", tau, b0, b1
    if tau > x.max(): return "p* reached only by extrapolation above observed scores", tau, b0, b1
    return "ok", max(tau, MIN_CONF), b0, b1

def evaluate(p_star, m):
    rows = []
    for r in sp.itertuples():
        code = r._1
        g = det[det.ebird_code == code]
        n, ntp = len(g), int(g.tp.sum())
        row = dict(ebird_code=code, common_name=r._3, n_annot_windows=len(ann_windows.get(code, ())),
                   n_detections=n, n_true_pos=ntp, precision_at_floor=(ntp / n if n else np.nan))
        if n == 0:
            row.update(status="not detected", validated=False, tau=np.nan)
        elif n < m:
            row.update(status=f"fewer than m={m} verified clips", validated=False, tau=np.nan)
        else:
            status, tau, b0, b1 = fit_threshold(g, p_star)
            ok = status.startswith("ok")
            row.update(status=status, validated=ok, tau=(tau if ok else np.nan), b0=b0, b1=b1)
            if ok:
                kept = g[g.confidence >= tau]
                row["n_kept"] = len(kept); row["precision_kept"] = kept.tp.mean() if len(kept) else np.nan
                aw = ann_windows.get(code, set())
                hit = set(zip(kept.filename, kept.win)) & aw
                row["recall_windows"] = len(hit) / len(aw) if aw else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    used = t[t.n_detections > 0]
    val = used[used.validated]
    summ = dict(p_star=p_star, m=m, n_target_species=len(t), n_used=len(used), n_validated=len(val),
                VTC=round(len(val) / len(used), 3),
                VTC_detection_weighted=round(val.n_detections.sum() / used.n_detections.sum(), 3),
                n_fail_too_few_clips=int(used.status.str.startswith("fewer").sum()),
                n_fail_p_not_reached=int(used.status.str.startswith("p* not reached").sum()),
                n_fail_extrapolation=int(used.status.str.contains("extrapolation").sum()),
                n_fail_other=int((~used.validated & ~used.status.str.startswith("fewer")
                                  & ~used.status.str.startswith("p* not reached")
                                  & ~used.status.str.contains("extrapolation")).sum()),
                median_precision_kept_validated=round(float(val.precision_kept.median()), 3) if len(val) else None,
                min_precision_kept_validated=round(float(val.precision_kept.min()), 3) if len(val) else None,
                validated_species=sorted(val.ebird_code.tolist()),
                median_tau=round(float(val.tau.median()), 3) if len(val) else None,
                tau_range=[round(float(val.tau.min()), 3), round(float(val.tau.max()), 3)] if len(val) else None,
                median_recall_validated=round(float(val.recall_windows.median()), 3) if len(val) else None,
                annotated_species_present=int((t.n_annot_windows > 0).sum()),
                annotated_species_among_validated=int((val.n_annot_windows > 0).sum()))
    return t, summ

t, headline = evaluate(P_STAR, M_CLIPS)
t.sort_values(["validated", "n_detections"], ascending=[False, False]).to_csv(os.path.join(out_dir, "vtc_species_table.csv"), index=False)
sens = [evaluate(p, m)[1] for p in (0.90, 0.95, 0.99) for m in (20, 50, 100)]
overall = dict(n_files=int(det.filename.nunique()), n_detections=len(det), n_true_pos=int(det.tp.sum()),
               precision_at_floor=round(float(det.tp.mean()), 3))
json.dump(dict(overall=overall, headline=headline, sensitivity=sens), open(os.path.join(out_dir, "vtc_summary.json"), "w"), indent=1)
print(json.dumps(dict(overall=overall, headline=headline), indent=1))
print(pd.DataFrame(sens)[["p_star", "m", "n_used", "n_validated", "VTC", "VTC_detection_weighted", "median_recall_validated"]].to_string(index=False))
print(t[["ebird_code", "n_annot_windows", "n_detections", "n_true_pos", "status", "tau", "n_kept", "precision_kept", "recall_windows"]].sort_values("n_detections", ascending=False).to_string(index=False))
