"""Studi C: bootstrap blok bergerak dan bootstrap klaster tahun untuk selisih macro-F1 terhadap klimatologi.
Pakai: python 33_bootstrap_blok_studiC.py <folder hasil>   (butuh _detail.csv dari 18b_keadaan_musim_simpan_detail.py)
Blok 10 harian berurutan waktu: 36 blok per tahun uji x 11 tahun uji (2015-2025). Bootstrap blok bergerak panjang 9 blok (kira-kira 3 bulan, lebih panjang dari
rentang prediksi 60 hari) dan bootstrap klaster tahun (11 tahun) menggantikan bootstrap baris, yang memperlakukan blok berdekatan sebagai independen."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
out = sys.argv[1]; d = pd.read_csv(f"{out}/_detail.csv"); rng = np.random.default_rng(1)
mf1 = lambda y, p: f1_score(y, p, average="macro", labels=[0, 1, 2]) if False else f1_score(y, p, average="macro")
MODELS = ["Persistence", "Pranata Mangsa (phase mode)", "Logistic (with calendar)", "LightGBM (with calendar)", "Logistic (weather only)", "LightGBM (weather only)"]
B = 1000; BL = 9; rows = []
for (site, lead), g in d.groupby(["site", "lead"]):
    base = g[g.model == "Climatology (10-day slot)"].reset_index(drop=True); n = len(base); yr = base.year.values
    for m in MODELS:
        gm = g[g.model == m].reset_index(drop=True); assert (gm.y.values == base.y.values).all()
        y, p, q = gm.y.values, gm.p.values, base.p.values; dobs = mf1(y, p) - mf1(y, q)
        # blok bergerak
        nb = int(np.ceil(n / BL)); v = []
        for _ in range(B):
            st = rng.integers(0, n - BL + 1, nb); s = np.concatenate([np.arange(a, a + BL) for a in st])[:n]; v.append(mf1(y[s], p[s]) - mf1(y[s], q[s]))
        lo_b, hi_b = np.percentile(v, [2.5, 97.5])
        # klaster tahun
        ys = np.unique(yr); idx = {k: np.flatnonzero(yr == k) for k in ys}; v = []
        for _ in range(B):
            s = np.concatenate([idx[k] for k in rng.choice(ys, len(ys))]); v.append(mf1(y[s], p[s]) - mf1(y[s], q[s]))
        lo_y, hi_y = np.percentile(v, [2.5, 97.5])
        # baris (acuan lama)
        v = []
        for _ in range(B):
            s = rng.integers(0, n, n); v.append(mf1(y[s], p[s]) - mf1(y[s], q[s]))
        lo_r, hi_r = np.percentile(v, [2.5, 97.5])
        rows.append(dict(site=site, lead=lead, model=m, n=n, dF1=dobs, lo_row=lo_r, hi_row=hi_r, lo_block=lo_b, hi_block=hi_b, lo_year=lo_y, hi_year=hi_y))
    print(site, lead, "selesai", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{out}/studiC_bootstrap_blok.csv", index=False)
R["sig_row"] = R.lo_row > 0; R["sig_block"] = R.lo_block > 0; R["sig_year"] = R.lo_year > 0
pd.set_option("display.width", 250); print(R.round(3).to_string())
