"""Step 4: revalidation after a model change. Compares VTC computed for two BirdNET versions.

Usage:  python compare_versions.py <results_dir_A> <label_A> <results_dir_B> <label_B> <out_dir>

Each results directory is the output of compute_vtc.py for one model version, run on the same recordings
with the same annotations and settings. The comparison shows what the revision rule in the manuscript
requires: thresholds are re-fitted for the new version, VTC is recomputed, and the change in the set of
validated species is reported, so that a change in method is not read as a change in the environment.
"""
import sys, os, json
import pandas as pd

dir_a, lab_a, dir_b, lab_b, out_dir = sys.argv[1:6]
os.makedirs(out_dir, exist_ok=True)

ta = pd.read_csv(os.path.join(dir_a, "vtc_species_table.csv"))
tb = pd.read_csv(os.path.join(dir_b, "vtc_species_table.csv"))
sa = json.load(open(os.path.join(dir_a, "vtc_summary.json")))
sb = json.load(open(os.path.join(dir_b, "vtc_summary.json")))

va, vb = set(ta.loc[ta.validated, "ebird_code"]), set(tb.loc[tb.validated, "ebird_code"])
both = sorted(va & vb)
m = ta.merge(tb, on="ebird_code", suffixes=(f"_{lab_a}", f"_{lab_b}"))
cols = ["ebird_code", "common_name_" + lab_a, f"n_detections_{lab_a}", f"n_detections_{lab_b}",
        f"status_{lab_a}", f"status_{lab_b}", f"tau_{lab_a}", f"tau_{lab_b}"]
m[cols].to_csv(os.path.join(out_dir, "version_comparison_species.csv"), index=False)

def head(s):
    h = s["headline"]
    return dict(n_detections=s["overall"]["n_detections"], precision_at_floor=s["overall"]["precision_at_floor"],
                n_used=h["n_used"], n_validated=h["n_validated"], VTC=h["VTC"],
                VTC_detection_weighted=h["VTC_detection_weighted"],
                median_recall_validated=h["median_recall_validated"])

tau_shift = (m.loc[m.ebird_code.isin(both), f"tau_{lab_b}"] - m.loc[m.ebird_code.isin(both), f"tau_{lab_a}"])
out = {
    lab_a: head(sa), lab_b: head(sb),
    "validated_in_both": both,
    f"validated_only_{lab_a}": sorted(va - vb),
    f"validated_only_{lab_b}": sorted(vb - va),
    "jaccard_validated_sets": round(len(va & vb) / len(va | vb), 3) if (va | vb) else None,
    "median_abs_tau_change_in_both": round(float(tau_shift.abs().median()), 3) if len(both) else None,
    "sensitivity": {lab_a: [(r["p_star"], r["m"], r["VTC"]) for r in sa["sensitivity"]],
                    lab_b: [(r["p_star"], r["m"], r["VTC"]) for r in sb["sensitivity"]]},
}
json.dump(out, open(os.path.join(out_dir, "version_comparison.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
