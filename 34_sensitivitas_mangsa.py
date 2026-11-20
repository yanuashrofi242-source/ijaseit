"""Sensitivitas Pranata Mangsa (Studi B): geser batas fase +-7 dan +-14 hari dan ganti hari acuan dalam bulan; AUC logistik rolling-origin Mangsa vs bulan kalender.
Pakai: python 34_sensitivitas_mangsa.py <folder kerja>. Butuh data_bersih/fitur_kalisat.csv."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
root = sys.argv[1]; rng = np.random.default_rng(2)
d = pd.read_csv(f"{root}/data_bersih/fitur_kalisat.csv", parse_dates=["Date", "target_date"]); d["bulan"] = d.Date.dt.month
START = [(6, 22), (8, 2), (8, 25), (9, 18), (10, 13), (11, 9), (12, 22), (2, 3), (3, 1), (3, 26), (4, 19), (5, 12)]   # Kasa ... Sadha, versi Surakarta
DOY = np.array([pd.Timestamp(2023, m, dd).dayofyear for m, dd in START])
def phase(ts, shift):
    st = (DOY + shift - 1) % 365 + 1; doy = np.minimum(ts.dt.dayofyear.values, 365)
    out = np.empty(len(ts), int)
    for i, x in enumerate(doy):
        diff = (x - st) % 365; out[i] = int(np.argmin(diff))      # fase yang dimulai paling dekat sebelum x
    return out
folds = [(d.target_date < "2023-01-01", (d.target_date >= "2023-01-01") & (d.target_date < "2024-01-01")),
         (d.target_date < "2024-01-01", (d.target_date >= "2024-01-01") & (d.target_date < "2025-01-01")), (d.target_date < "2025-01-01", d.target_date >= "2025-01-01")]
def cv(X, ycol):
    ys, ps = [], []
    for tri, tei in folds:
        m = LogisticRegression(C=0.5, max_iter=3000).fit(X[tri], d[ycol][tri]); ps.append(m.predict_proba(X[tei])[:, 1]); ys.append(d[ycol][tei].values)
    return np.concatenate(ys), np.concatenate(ps)
rows = []
for ycol, tn in (("y_next", "LKSJ>0 next month"), ("y_next_SSJ", "SSJ>0 next month (main)")):
    yb, pb = cv(pd.get_dummies(d["bulan"].astype(str)).astype(float), ycol); ab = roc_auc_score(yb, pb)
    for ref_day, ref in ((0, "day 1"), (14, "day 15"), (27, "day 28")):
        mid = d.Date + pd.Timedelta(days=ref_day)
        for sh in (-14, -7, 0, 7, 14):
            ph = phase(mid, sh); X = pd.get_dummies(pd.Series(ph).astype(str)).astype(float); X.index = d.index
            y, p = cv(X, ycol); a = roc_auc_score(y, p)
            rows.append(dict(target=tn, ref=ref, shift=sh, phases_used=len(np.unique(ph)), auc_mangsa=a, auc_month=ab, diff=a - ab))
R = pd.DataFrame(rows); R.to_csv(f"{root}/hasil/studiB_sensitivitas_mangsa.csv", index=False)
pd.set_option("display.width", 200); print(R.round(3).to_string())
print(R.groupby("target")["diff"].agg(["min", "max", "mean"]).round(3)); print(R.groupby("target")[["auc_mangsa"]].agg(["min", "max"]).round(3))
# selisih Mangsa (baseline: hari 15, geser 0) vs bulan: bootstrap klaster desa dan bulan
ph = phase(d.Date + pd.Timedelta(days=14), 0); X = pd.get_dummies(pd.Series(ph).astype(str)).astype(float); X.index = d.index
for ycol in ("y_next", "y_next_SSJ"):
    y, p = cv(X, ycol); yb, pb = cv(pd.get_dummies(d["bulan"].astype(str)).astype(float), ycol)
    grp_v = np.concatenate([d.Desa[te].values for _, te in folds]); grp_m = np.concatenate([d.target_date[te].values.astype("datetime64[M]").astype(str) for _, te in folds])
    for nm, grp in (("village", grp_v), ("month", grp_m)):
        u = np.unique(grp); idx = {g: np.flatnonzero(grp == g) for g in u}; v = []
        for _ in range(1000):
            s = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
            if y[s].min() == y[s].max(): continue
            v.append(roc_auc_score(y[s], p[s]) - roc_auc_score(y[s], pb[s]))
        print(ycol, nm, "cluster bootstrap AUC(Mangsa) - AUC(month):", round(roc_auc_score(y, p) - roc_auc_score(y, pb), 3), np.round(np.percentile(v, [2.5, 97.5]), 3))
