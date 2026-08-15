import csv, math

ROOT = "."
def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))
def num(x):
    x = str(x).strip().replace(",", "")
    if x in ("", "nan", "inf", "-inf"): return None
    return float(x)

results = []
def check(table, cell, printed, computed, tol=1e-3):
    if printed is None or computed is None:
        ok = printed == computed
    else:
        s = f"{printed:g}"
        dec = len(s.split('.')[1]) if '.' in s else 0
        scale = 10 ** -dec
        ok = abs(printed - computed) <= max(tol, scale * 0.55)
    results.append((table, cell, printed, computed, ok))
    if not ok:
        print(f"  FAIL {table} {cell}: printed={printed}  csv={computed}")

# ---------------- tab:cp (Table 5) ----------------
# Source: paper_cp_summary.csv. Scenario mapping:
#   Within -> within_split_cp (adapter none)
#   Naive source-cal -> cross_source_calibrated_cp (adapter none)
#   Target-domain (k=20) -> cross_target_adapted_cp with adapter_type? need to inspect
#   Residual/Linear-adapted -> cross_target_adapted_cp (adapter residual_mean / linear)
cp = load(f"{ROOT}/outputs/results_v2_four_dataset_conformal/paper_cp_summary.csv")
print("== tab:cp (Table 5) ==")

def cp_row(scenario, adapter, src, tgt, conf, kt=None):
    for r in cp:
        if (r["scenario"]==scenario and r.get("adapter_type","")==adapter
                and r["source"]==src and r["target"]==tgt
                and r["confidence_level"]==conf and r["n_cycles"]=="100"):
            if kt is not None and r.get("k_target","") != kt:
                continue
            return r
    return None

# Printed rows (scenario, alpha, direction, model, Cov, Fin, W, MAE, sMAPE, R2)
cp_printed = [
    # Within rows
    ("within_split_cp", "none", "0.9", "matr", "matr", "catboost", 0.910, 1.00, 1053, 193, 23.2, 0.43),
    ("within_split_cp", "none", "0.9", "hust", "hust", "random_forest", 0.967, 1.00, 1056, 209, 14.1, 0.12),
    ("within_split_cp", "none", "0.9", "sandia", "sandia", "xgboost", 0.875, 1.00, 762, 130, 24.9, 0.96),
    ("within_split_cp", "none", "0.9", "luh", "luh", "gaussian_process", 0.900, 1.00, 623, 116, 18.1, 0.78),
    # Naive source-cal
    ("cross_source_calibrated_cp", "none", "0.9", "matr", "hust", "catboost", 0.177, 1.00, 1053, 909, 85.8, -11.09),
    ("cross_source_calibrated_cp", "none", "0.9", "hust", "matr", "random_forest", 0.203, 1.00, 1056, 793, 73.3, -4.52),
    ("cross_source_calibrated_cp", "none", "0.9", "sandia", "matr", "xgboost", 0.107, 1.00, 762, 1071, 83.0, -9.65),
    ("cross_source_calibrated_cp", "none", "0.9", "luh", "hust", "gaussian_process", 0.000, 1.00, 623, 1084, 112.4, -15.63),
    # Target-domain (k=20) = cross_target_calibrated_cp (adapter none, k_target=20)
    ("cross_target_calibrated_cp", "none", "0.9", "matr", "hust", "catboost", 0.902, 1.00, 2531, 897, 85.1, -10.50),
    ("cross_target_calibrated_cp", "none", "0.9", "hust", "matr", "random_forest", 0.914, 1.00, 2315, 793, 73.3, -4.44),
    ("cross_target_calibrated_cp", "none", "0.9", "sandia", "luh", "xgboost", 0.913, 1.00, 1872, 323, 57.9, -0.33),
    # Residual-adapted (k=20+20)
    ("cross_target_adapted_cp", "residual_mean", "0.9", "matr", "hust", "catboost", 0.906, 1.00, 1008, 246, 16.7, -0.17),
    ("cross_target_adapted_cp", "residual_mean", "0.9", "hust", "matr", "random_forest", 0.915, 1.00, 1303, 269, 35.2, 0.05),
    ("cross_target_adapted_cp", "residual_mean", "0.9", "sandia", "luh", "xgboost", 0.905, 1.00, 1314, 291, 59.5, 0.22),
    # Linear-adapted
    ("cross_target_adapted_cp", "linear", "0.9", "sandia", "luh", "xgboost", 0.903, 1.00, 1098, 212, 54.3, 0.54),
    ("cross_target_adapted_cp", "linear", "0.9", "hust", "matr", "random_forest", 0.896, 1.00, 1321, 277, 36.6, 0.02),
    # 0.95 residual-adapted
    ("cross_target_adapted_cp", "residual_mean", "0.95", "matr", "hust", "catboost", 0.953, 1.00, 1164, 246, 16.7, -0.17),
    ("cross_target_adapted_cp", "residual_mean", "0.95", "hust", "matr", "random_forest", 0.962, 1.00, 1822, 269, 35.2, 0.05),
]
for (scen, adap, conf, src, tgt, model, cov, fin, w, mae, smape, r2) in cp_printed:
    kt = "20.0" if scen == "cross_target_calibrated_cp" else None
    row = cp_row(scen, adap, src, tgt, conf, kt)
    if not row:
        print(f"  MISSING cp row {scen}/{adap} {src}->{tgt} {conf}")
        continue
    label = f"{scen}/{adap} {src}->{tgt} {conf}"
    check("tab:cp", label+" cov", cov, num(row["coverage_mean"]))
    check("tab:cp", label+" fin", fin, num(row["finite_interval_fraction_mean"]))
    check("tab:cp", label+" W", w, num(row["median_width_mean"]))
    check("tab:cp", label+" MAE", mae, num(row["MAE_median"]))
    check("tab:cp", label+" sMAPE", smape, num(row["SMAPE_median"]))
    check("tab:cp", label+" R2", r2, num(row["R2_median"]))

# ---------------- tab:t1parity (Table 6) ----------------
t1 = load(f"{ROOT}/outputs/results_v2_four_dataset_conformal_t1/results_summary.csv")
print("== tab:t1parity (Table 6) ==")
# Adapted = residual-mean adapted (cross_target_adapted_cp/residual_mean); Target-only = cross_target_only_cp
# Printed: direction, adapted MAE, target-only MAE, delta
t1_printed = {
    ("hust","luh"):    (370.2, 148.6, +221.5),
    ("hust","matr"):   (268.6, 259.8, +8.7),
    ("hust","sandia"): (772.1, 272.8, +499.4),
    ("luh","hust"):    (232.1, 235.6, -3.4),
    ("luh","matr"):    (281.7, 267.6, +14.1),
    ("luh","sandia"):  (823.6, 298.4, +525.2),
    ("matr","hust"):   (245.7, 245.1, +0.5),
    ("matr","luh"):    (329.6, 153.4, +176.2),
    ("matr","sandia"): (816.8, 269.5, +547.3),
    ("sandia","hust"): (686.7, 242.4, +444.3),
    ("sandia","luh"):  (290.7, 150.9, +139.8),
    ("sandia","matr"): (384.6, 263.2, +121.4),
}
for (src, tgt), (adapted, tgtonly, delta) in t1_printed.items():
    arow = next((r for r in t1 if r["scenario"]=="cross_target_adapted_cp"
                 and r["adapter_type"]=="residual_mean" and r["source"]==src and r["target"]==tgt
                 and r["confidence_level"]=="0.9" and r["n_cycles"]=="100"
                 and r["k_adapter"]=="20.0" and r["k_target"]=="20.0"), None)
    trow = next((r for r in t1 if r["scenario"]=="cross_target_only_cp"
                 and r["source"]==src and r["target"]==tgt
                 and r["confidence_level"]=="0.9" and r["n_cycles"]=="100"
                 and r["k_adapter"]=="20.0" and r["k_target"]=="20.0"), None)
    if not arow or not trow:
        print(f"  MISSING t1 row {src}->{tgt} (adapted={bool(arow)} targetonly={bool(trow)})")
        continue
    label = f"{src}->{tgt}"
    check("tab:t1parity", label+" adapted", adapted, num(arow["MAE_median"]))
    check("tab:t1parity", label+" targetonly", tgtonly, num(trow["MAE_median"]))
    # delta = median over runs of (adapted - targetonly)? compute from CSV if available; else check printed delta vs arow/trow median difference
    check("tab:t1parity", label+" delta", delta, num(arow["MAE_median"]) - num(trow["MAE_median"]))

fails = [r for r in results if not r[4]]
print()
print(f"TOTAL CHECKS: {len(results)}  PASS: {len(results)-len(fails)}  FAIL: {len(fails)}")
for t,c,p,v,ok in fails:
    print(f"  FAIL {t} {c}: printed={p} csv={v}")
