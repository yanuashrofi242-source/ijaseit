"""Studi A: semua metode pada hari-hari yang sama (sampel bersama) dan koreksi Holm.
Sampel bersama = hari uji yang punya jendela 14 hari lengkap untuk jaringan saraf (sama dengan 16_nn_pembanding.py): 473 (lead 1), 494 (lead 3), 487 (lead 7).
Pakai: python 35_sampel_bersama_holm.py <data_bersih> <hasil>"""
import sys, pickle, numpy as np, pandas as pd
from scipy import stats
dd, out = sys.argv[1], sys.argv[2]
SITES = {"Kalisat": ("nwp_arsip_ecmwf_ifs025_kalisat.csv", "om_harian_kalisat_2005_sekarang.csv"), "Surabaya": ("nwp_arsip_ecmwf_ifs025_sensor.csv", "om_harian_sensor_2005_sekarang.csv")}
V = {"suhu": ("temperature_2m", "temperature_2m_mean", "C"), "rh": ("relative_humidity_2m", "relative_humidity_2m_mean", "%"), "angin": ("wind_speed_10m", "wind_speed_10m_mean", "km/h"), "hujan": ("precipitation", "precipitation_sum", "mm")}
TV = ["suhu", "rh", "angin"]; LEADS = (1, 3, 7); WIN_DAYS = 14
def daily_nwp(f):
    h = pd.read_csv(f, parse_dates=["time"]); h["d"] = h.time.dt.floor("D"); o = {}
    for k, (b, _, _) in V.items():
        for L in LEADS:
            a = h.groupby("d")[f"{b}_previous_day{L}"]; o[f"{k}_n{L}"] = a.sum(min_count=24) if k == "hujan" else a.mean().where(a.count() >= 24)
    return pd.DataFrame(o)
def dm(e1, e2, h):
    d = np.abs(e1) - np.abs(e2); n = len(d); dm_ = d - d.mean(); s = (dm_ ** 2).mean()
    for k in range(1, max(h, 1)): s += 2 * (1 - k / max(h, 1)) * (dm_[k:] * dm_[:-k]).mean()
    stat = d.mean() / np.sqrt(max(s, 1e-12) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return d.mean(), 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
P = pickle.load(open(f"{out}/_pred.pkl", "rb")); rows = []; MAES = []
for site, (fn, ft) in SITES.items():
    tr_ = pd.read_csv(f"{dd}/{ft}", parse_dates=["time"]).set_index("time").sort_index(); nw = daily_nwp(f"{dd}/{fn}"); D = nw.join(tr_[[c for _, c, _ in V.values()]], how="inner")
    for L in LEADS:
        T = pd.DataFrame({k: D[V[k][1]] for k in V}); N = pd.DataFrame({k: D[f"{k}_n{L}"] for k in V}); R = (T - N)[TV]
        feat = pd.concat([T[TV], np.log1p(T[["hujan"]].clip(lower=0)), R.add_suffix("_res")], axis=1); fv = feat.values; idx = D.index; N_ = len(idx); ok = np.zeros(N_, bool)
        for i in range(N_):
            e = i - L
            if e - WIN_DAYS + 1 < 0: continue
            w = fv[e - WIN_DAYS + 1: e + 1]; z = [N.suhu.iloc[i], N.rh.iloc[i], N.angin.iloc[i], N.hujan.iloc[i]]; y = R.iloc[i].values
            ok[i] = not (np.isnan(w).any() or np.isnan(z).any() or np.isnan(y).any())
        for k in TV:
            dates, yt, M = P[(site, k, L)]; pos = idx.get_indexer(pd.DatetimeIndex(dates)); sel = np.flatnonzero(ok[pos]); y = yt[sel]
            Mm = {m: np.asarray(v)[sel] for m, v in M.items()}
            mae = {m: np.mean(np.abs(y - v)) for m, v in Mm.items()}
            MAES.append(dict(site=site, var=k, L=L, n=len(sel), **{f"mae {m}": v for m, v in mae.items()}))
            for a, b, fam in (("Ridge walk-forward", "Bias bergulir 30 hari", "Ridge vs rolling bias"), ("Ridge walk-forward", "NWP mentah", "Ridge vs raw NWP"), ("Bias bergulir 30 hari", "NWP mentah", "Rolling bias vs raw NWP"),
                              ("LightGBM residu walk-forward", "Bias bergulir 30 hari", "LightGBM vs rolling bias"), ("Ridge walk-forward", "Persistensi", "Ridge vs persistence")):
                d_, p_ = dm(y - Mm[a], y - Mm[b], L); rows.append(dict(family=fam, site=site, var=k, L=L, n=len(sel), d=d_, p=p_))
MA = pd.DataFrame(MAES); MA.to_csv(f"{out}/studiA_sampel_bersama_mae.csv", index=False)
nn = pd.read_csv(f"{out}/studiA_nn_hasil.csv")
for _, r in nn.iterrows():
    rows.append(dict(family="BiGRU vs Ridge", site=r.site, var=r["var"], L=r.L, n=r.n, d=r["mae BiGRU-Attention kuantil (median)"] - r["mae Ridge walk-forward"], p=r.p_dm_vs_ridge))
    rows.append(dict(family="BiGRU vs rolling bias", site=r.site, var=r["var"], L=r.L, n=r.n, d=r["mae BiGRU-Attention kuantil (median)"] - r["mae Bias bergulir 30 hari"], p=r.p_dm_vs_bias))
for fn, nm in (("studiA_nbeats_hasil.csv", "N-BEATS-64 vs Ridge"), ("studiA_nbeats_kecil_hasil.csv", "N-BEATS-16 vs Ridge")):
    nb = pd.read_csv(f"{out}/{fn}"); mc = [c for c in nb.columns if c.startswith("mae") and "N-BEATS" in c][0]
    for _, r in nb.iterrows(): rows.append(dict(family=nm, site=r.site, var=r["var"], L=r.L, n=r.n, d=r[mc] - r["mae Ridge walk-forward"], p=r.p_dm_vs_ridge))
Rr = pd.DataFrame(rows); Rr["p_holm"] = np.nan
for f, g in Rr.groupby("family", sort=False): Rr.loc[g.index, "p_holm"] = holm(g.p.values)
Rr.to_csv(f"{out}/studiA_holm_sampel_bersama.csv", index=False)
S = pd.DataFrame([dict(family=f, cells=len(g), better=int((g.d < 0).sum()), raw_better=int(((g.p < .05) & (g.d < 0)).sum()), raw_worse=int(((g.p < .05) & (g.d > 0)).sum()), holm_better=int(((g.p_holm < .05) & (g.d < 0)).sum()), holm_worse=int(((g.p_holm < .05) & (g.d > 0)).sum())) for f, g in Rr.groupby("family", sort=False)])
S.to_csv(f"{out}/studiA_holm_ringkasan_bersama.csv", index=False); pd.set_option("display.width", 250); print(S.to_string()); print(MA.groupby("L").n.unique())
x = Rr[(Rr.family == "Ridge vs rolling bias")]; print(x.round(4).to_string())
mk = lambda g: g.mean()
