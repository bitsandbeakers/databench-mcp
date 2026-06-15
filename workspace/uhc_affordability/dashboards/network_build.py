"""
One-off build: materialize the hospital service-mix similarity network for the dashboard.

Reuses the canonical Louvain community assignments (hosp_community) and archetype names
(archetype_map) from the analysis — does NOT recompute the partition. Computes cosine-kNN
edges and a seeded Fruchterman-Reingold layout over a stratified sample, then writes
nodes.parquet + edges.parquet into ./data for the Dash app to render.

Run:  uv run --with pyarrow python network_build.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DB = HERE.parent / "project.duckdb"
OUT = HERE / "data"
SEED = 7
SAMPLE = 800          # nodes drawn (full network is 3,236; sampled for legibility)
K = 8                 # cosine kNN per node

# 11 high-confidence archetype-adjusted steerage targets (h013) -> short labels
TARGETS = {
    "310092": "Capital Health Regional", "310016": "Carepoint Christ",
    "310025": "Carepoint Bayonne", "310040": "Carepoint Hoboken",
    "310044": "Capital Health Hopewell", "050125": "RMC San Jose",
    "050779": "MLK Jr Community", "670280": "N. Houston Surgical",
    "050380": "Good Samaritan (CA)", "050441": "Stanford", "060014": "Presby/St Luke's",
}

rng = np.random.default_rng(SEED)

con = duckdb.connect(str(DB), read_only=True)
mat = con.execute("SELECT * FROM hosp_svc_matrix").df()
comm = con.execute("SELECT ccn, community FROM hosp_community").df()
amap = con.execute("SELECT community, archetype, archetype_group FROM archetype_map").df()
con.close()

# join community + archetype labels onto the feature matrix
df = mat.merge(comm, on="ccn", how="inner").merge(amap, on="community", how="left")
feat_cols = [c for c in mat.columns if c != "ccn"]
df = df.dropna(subset=["archetype_group"]).reset_index(drop=True)

# stratified sample by community (keep small communities whole)
parts = []
for c, g in df.groupby("community"):
    take = min(len(g), max(12, round(SAMPLE * len(g) / len(df))))
    parts.append(g.sample(n=take, random_state=SEED) if len(g) > take else g)
s = pd.concat(parts).reset_index(drop=True)

# force-include the 11 steerage targets so they always appear on the graph
miss = df[df["ccn"].isin(TARGETS) & ~df["ccn"].isin(s["ccn"])]
s = pd.concat([s, miss]).drop_duplicates(subset="ccn").reset_index(drop=True)

X = s[feat_cols].to_numpy(dtype=float)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)   # L2 -> cosine via euclidean
n = len(s)

# cosine kNN edges (undirected, deduped)
nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(X)
_, idx = nn.kneighbors(X)
eset = set()
for i in range(n):
    for j in idx[i, 1:]:
        eset.add((min(i, int(j)), max(i, int(j))))
edges = np.array(sorted(eset))

# ---- seeded Fruchterman-Reingold layout (PCA init) ----
# PCA(2) init for a stable, reproducible starting point
Xc = X - X.mean(0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
pos = Xc @ Vt[:2].T
pos = (pos - pos.mean(0)) / (pos.std(0) + 1e-9)
pos += rng.normal(0, 0.01, pos.shape)

ei, ej = edges[:, 0], edges[:, 1]
k = 1.0 / np.sqrt(n)
t = 0.10
for _ in range(300):
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9
    rep = ((k * k) / dist)[..., None] * (delta / dist[..., None])
    disp = rep.sum(axis=1)
    d = pos[ei] - pos[ej]
    dd = np.sqrt((d ** 2).sum(-1)) + 1e-9
    att = ((dd * dd) / k)[..., None] * (d / dd[..., None])
    np.add.at(disp, ei, -att)
    np.add.at(disp, ej, att)
    dlen = np.sqrt((disp ** 2).sum(-1)) + 1e-9
    pos += (disp / dlen[..., None]) * np.minimum(dlen, t)[..., None]
    t *= 0.985

# normalize to a tidy frame
pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0) + 1e-9) * 2 - 1

nodes = pd.DataFrame({
    "x": pos[:, 0], "y": pos[:, 1],
    "ccn": s["ccn"].values,
    "community": s["community"].values,
    "archetype": s["archetype"].values,
    "archetype_group": s["archetype_group"].values,
})
nodes["is_target"] = nodes["ccn"].isin(TARGETS)
nodes["label"] = nodes["ccn"].map(TARGETS).fillna("")

# precompute edge segment coordinates (x0,y0,x1,y1) for fast plotting
edf = pd.DataFrame({
    "x0": pos[ei, 0], "y0": pos[ei, 1],
    "x1": pos[ej, 0], "y1": pos[ej, 1],
})

OUT.mkdir(exist_ok=True)
nodes.to_parquet(OUT / "net_nodes.parquet")
edf.to_parquet(OUT / "net_edges.parquet")
print(f"nodes={len(nodes)} edges={len(edf)} communities={nodes['community'].nunique()} "
      f"groups={nodes['archetype_group'].nunique()} targets={int(nodes['is_target'].sum())}")
print("wrote net_nodes.parquet, net_edges.parquet")
