"""
Standalone findings dashboard (PNG) for the provider cost-driver analysis.
Renders the already-computed results (no MCP needed). 6 panels.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False   # treat $ as a literal dollar sign
import matplotlib.pyplot as plt
import numpy as np

FP = "#c0392b"      # for-profit / excess (red)
NP = "#2e86c1"      # comparison (blue)
GY = "#7f8c8d"

fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
fig.suptitle("UHC Affordability — Hospital Cost-Driver Dashboard (2023 CMS Medicare inpatient)",
             fontsize=15, fontweight="bold")

# 1. Triangulated importance ranking
feats = ["state", "case-mix", "ownership", "n_drgs", "urbanicity", "archetype"]
shap_ = [0.302, 0.251, 0.210, 0.085, 0.051, 0.066]
perm_ = [0.279, 0.603, 0.247, 0.040, 0.017, 0.013]
ebm_  = [0.202, 0.222, 0.145, 0.061, 0.054, 0.038]
norm = lambda v: np.array(v) / np.sum(v)
x = np.arange(len(feats)); w = 0.27
a = ax[0, 0]
a.bar(x - w, norm(shap_), w, label="SHAP", color="#34495e")
a.bar(x,     norm(perm_), w, label="Permutation", color="#16a085")
a.bar(x + w, norm(ebm_),  w, label="EBM", color="#e67e22")
a.set_xticks(x); a.set_xticklabels(feats, rotation=25, ha="right", fontsize=9)
a.set_title("1. Driver importance — 3 methods agree\n(top trio: geography, case-mix, ownership)", fontsize=11)
a.set_ylabel("share of importance"); a.legend(fontsize=8)

# 2. For-profit premium survives controls
stages = ["raw", "+case-mix", "+archetype", "+urbanicity", "+state"]
prem = [62.5, 55.2, 55.9, 52.9, 46.3]
a = ax[0, 1]
a.plot(stages, prem, "-o", color=FP, lw=2)
a.fill_between(range(len(stages)), prem, alpha=0.12, color=FP)
for i, v in enumerate(prem):
    a.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=9, color=FP)
a.set_ylim(0, 72); a.set_xticks(range(len(stages)))
a.set_xticklabels(stages, rotation=25, ha="right", fontsize=9)
a.set_title("2. For-profit premium survives every control\n(+62.5% raw → +46.3% fully adjusted)", fontsize=11)
a.set_ylabel("for-profit charge premium")

# 3. Premium by archetype
arch = ["Generalist", "Rehab-Spec", "Academic", "Rural-SmAcute", "Surgical-Spec"]
av = [64.6, 50.4, 46.7, 22.8, -11.0]
a = ax[0, 2]
cols = [FP if v > 0 else NP for v in av]
a.barh(arch[::-1], av[::-1], color=cols[::-1])
a.axvline(0, color="k", lw=0.8)
for i, v in enumerate(av[::-1]):
    a.text(v + (1 if v >= 0 else -1), i, f"{v:+.0f}%", va="center",
           ha="left" if v >= 0 else "right", fontsize=9)
a.set_title("3. Premium concentrates in steerable\ngeneralist care (surgical niche charges less)", fontsize=11)
a.set_xlabel("for-profit premium")

# 4. Charge vs payment falsification
a = ax[1, 0]
bars = a.bar(["Submitted\ncharge", "Medicare\npayment"], [45.4, -8.9], color=[FP, NP])
a.axhline(0, color="k", lw=0.8)
for b, v in zip(bars, [45.4, -8.9]):
    a.text(b.get_x() + b.get_width()/2, v + (2 if v >= 0 else -3),
           f"{v:+.0f}%", ha="center", fontsize=11, fontweight="bold")
a.set_ylim(-20, 55)
a.set_title("4. Falsification: pricing, not cost\n(premium on charges, gone on Medicare payment)", fontsize=11)
a.set_ylabel("for-profit premium (full controls)")

# 5. Dollar sizing / steerage
a = ax[1, 1]
labels = ["FP billed", "Excess\n(premium)", "25%\nsteer", "50%\nsteer", "100%\nsteer"]
vals = [81.6, 25.8, 6.46, 12.92, 25.84]
cols = [GY, FP, "#27ae60", "#27ae60", "#27ae60"]
bars = a.bar(labels, vals, color=cols)
for b, v in zip(bars, vals):
    a.text(b.get_x() + b.get_width()/2, v + 1.2, f"${v:.1f}B", ha="center", fontsize=9)
a.set_title("5. Dollar prize ($B, Medicare inpatient)\n~$25.8B for-profit premium; 50% steer ≈ $12.9B", fontsize=11)
a.set_ylabel("$ billions")

# 6. h011 rural Medicare exposure
a = ax[1, 2]
grp = ["R-CAH", "R-PPS", "U-CAH", "U-PPS"]
mcr = [48.3, 27.6, 43.0, 24.3]
negop = [67.5, 61.5, 53.3, 48.4]
x = np.arange(len(grp)); w = 0.38
a.bar(x - w/2, mcr, w, label="Medicare day share %", color="#8e44ad")
a.bar(x + w/2, negop, w, label="% with neg. op. margin", color="#d35400")
a.set_xticks(x); a.set_xticklabels(grp, fontsize=9)
a.set_title("6. h011 — rural Medicare exposure\n(highest dependence + no margin cushion)", fontsize=11)
a.set_ylabel("percent"); a.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = r"C:\Users\cody\nox-code\satellites\databench-mcp\provider_drivers_dashboard.png"
fig.savefig(out, dpi=130)
print("saved:", out)
