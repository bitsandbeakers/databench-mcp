"""
UHC Affordability — narrative dashboard.

A story-led view of the 2023 CMS Medicare hospital-pricing analysis (18 hypotheses,
all supported). Main flow reads top-to-bottom for a mixed technical audience; methods,
model metrics, sources, and caveats live in the collapsible appendix at the bottom.

Run:
    pip install dash plotly pandas pyarrow
    python app.py            # serves at http://127.0.0.1:8050
"""
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- palette (Optum — see design.md)
# Official Optum data-visualization palette (brand.optum.com/content/data-visualization).
# Charts use THIS palette, not the brand orange (Optum keeps data colors separate from brand).
DV_TURQUOISE = "#15A796"
DV_PINK      = "#C72887"
DV_PURPLE    = "#8061BC"
DV_TANGERINE = "#E4780C"
DV_SAPPHIRE  = "#1E82CB"
# role aliases
FP   = DV_TANGERINE   # for-profit / excess / problem
CMP  = DV_SAPPHIRE    # comparison / regulated
OK   = DV_TURQUOISE   # opportunity / steerable
BRIGHT   = DV_PINK    # secondary series (e.g. outpatient)
MARIGOLD = DV_PURPLE  # third / accent series
INK   = "#3D3C38"     # Warm Gray (official text color) — reference lines, axis text
MUTE  = "#6e6c66"     # lightened warm gray
GRAY2 = "#8a887f"
GRID  = "#EDE8E0"     # Horizon — gridlines
# single-hue Tangerine sequential ramp for choropleth / intensity
TANGERINE_SCALE = ["#FBE7CE", "#F4B26B", "#E4780C", "#A6560A"]
# archetype-group colors for the service-mix network (7 groups)
GROUP_COLORS = {
    "Generalist":            DV_SAPPHIRE,
    "Academic-Tertiary":     DV_PINK,
    "Surgical-Specialty":    DV_PURPLE,
    "Rural-SmallAcute":      DV_TURQUOISE,
    "Rehab-Specialty":       DV_TANGERINE,
    "Rehab-Regulated":       "#6e6c66",
    "Specialty-Behavioral":  "#0f7c6e",
}

# ---------------------------------------------------------------- data
def _coerce(df):
    # DuckDB exports DECIMAL columns as Python Decimal objects (dtype=object),
    # which plotly/JSON cannot serialize. Coerce numeric-looking object cols to float.
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().any() and conv.notna().sum() >= df[c].notna().sum():
                df[c] = conv
    return df


t = {f.stem: _coerce(pd.read_parquet(f)) for f in DATA.glob("*.parquet")}


def style(fig, h=360, legend=True, lock=True):
    fig.update_layout(
        template="plotly_white",
        height=h,
        margin=dict(l=10, r=18, t=14, b=10),
        font=dict(family="Enterprise Sans, Arial, system-ui, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
    )
    # lock=True disables zoom/pan (keeps hover + legend-click); only the network passes lock=False.
    # fixedrange handles cartesian; dragmode=False also covers the geo choropleth (no cartesian axes).
    if lock:
        fig.update_layout(dragmode=False)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, fixedrange=lock)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, fixedrange=lock)
    return fig


# ================================================================ FIGURES

# --- h001: skew of submitted charges (log) ---
def fig_skew():
    df = t["inpatient_enr"]
    col = "Avg_Submtd_Cvrd_Chrg"
    x = np.log10(df[col].replace(0, np.nan).dropna())
    fig = go.Figure(go.Histogram(x=x, nbinsx=70, marker_color=FP, opacity=0.85))
    med = np.log10(df[col].median())
    fig.add_vline(x=med, line_dash="dash", line_color=INK,
                  annotation_text=f"median ${df[col].median():,.0f}", annotation_position="top right")
    fig.update_xaxes(title="submitted charge per provider-DRG (log₁₀ $)")
    fig.update_yaxes(title="provider-DRG rows")
    return style(fig, 360, legend=False)


# --- h002: same service, different price (box, log) ---
def fig_dispersion():
    df = t["chart_h002_box"].copy()
    order = df.groupby("drg_label")["charge"].median().sort_values().index.tolist()
    fig = px.box(df, x="drg_label", y="charge", color="drg_label",
                 category_orders={"drg_label": order}, points=False,
                 color_discrete_sequence=["#FBE7CE", "#F4B26B", "#EE9A3C", "#E4780C", "#C2650A", "#A6560A"])
    fig.update_yaxes(type="log", title="submitted charge ($, log)")
    fig.update_xaxes(title="")
    return style(fig, 380, legend=False)


# --- h003: highest-markup hospitals ---
def fig_markup():
    df = t["chart_h003_top"].sort_values("med_markup")
    fig = go.Figure(go.Bar(
        x=df["med_markup"], y=df["hospital_label"], orientation="h",
        marker_color=FP, text=[f"{v:.1f}×" for v in df["med_markup"]],
        textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="median charge ÷ Medicare payment")
    fig.update_yaxes(title="")
    return style(fig, 460, legend=False)


# --- h014: pricing culture IP vs OP ---
def fig_pricing_culture():
    df = t["prov_pricing_idx"].dropna(subset=["ip_charge_idx", "op_charge_idx"])
    df = df[(df["ip_charge_idx"] > 0) & (df["op_charge_idx"] > 0)]
    fig = go.Figure(go.Scattergl(
        x=df["ip_charge_idx"], y=df["op_charge_idx"], mode="markers",
        marker=dict(size=5, color=FP, opacity=0.35), name="hospital",
        hovertemplate="IP %{x:.2f} · OP %{y:.2f}<extra></extra>"))
    # no OLS fit — the indices are right-skewed and the relationship is summarized by the
    # rank (Spearman) correlation; show log-log axes + an equal-pricing reference instead
    lo = min(df["ip_charge_idx"].min(), df["op_charge_idx"].min())
    hi = max(df["ip_charge_idx"].max(), df["op_charge_idx"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=INK, width=1.5, dash="dash"), name="equal IP = OP"))
    fig.add_annotation(xref="paper", yref="paper", x=0.03, y=0.97, showarrow=False,
                       text="Spearman ρ = 0.82 (rank)", font=dict(size=12, color=INK),
                       align="left", bgcolor="rgba(255,255,255,0.7)")
    fig.update_xaxes(type="log", title="inpatient charge index (case-mix adj., log)")
    fig.update_yaxes(type="log", title="outpatient charge index (log)")
    return style(fig, 360)


# --- h015: falsification charge vs payment ---
def fig_falsification():
    fig = go.Figure(go.Bar(
        x=["Submitted charge", "Medicare payment"], y=[45.4, -8.9],
        marker_color=[FP, CMP], text=["+45%", "−9%"], textposition="outside",
        cliponaxis=False))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    fig.update_yaxes(title="for-profit premium (full controls, %)", range=[-20, 58])
    return style(fig, 360, legend=False)


# --- h015: premium survives controls (discrete control sets -> bars, not a line) ---
def fig_premium_stages():
    stages = ["raw", "+case-mix", "+archetype", "+urbanicity", "+state<br>(full)"]
    prem = [62.5, 55.2, 55.9, 52.9, 46.3]
    fig = go.Figure(go.Bar(
        x=stages, y=prem, marker_color=FP,
        text=[f"{v:.0f}%" for v in prem], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="for-profit charge premium (%)", range=[0, 72])
    fig.update_xaxes(title="controls included")
    return style(fig, 360, legend=False)


# --- h004: state choropleth ---
def fig_states():
    df = t["state_cost_index"]
    fig = px.choropleth(df, locations="state", locationmode="USA-states",
                        color="adj_charge_idx", scope="usa",
                        color_continuous_scale=TANGERINE_SCALE,
                        labels={"adj_charge_idx": "charge index"})
    fig.update_layout(coloraxis_colorbar=dict(title="index", thickness=12, len=0.8))
    return style(fig, 380, legend=False)


# --- h010: urbanicity ---
def fig_urbanicity():
    df = t["chart_h010"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["urbanicity"], y=df["inpatient_rel_charge"],
                         name="Inpatient", marker_color=FP))
    fig.add_trace(go.Bar(x=df["urbanicity"], y=df["outpatient_rel_charge"],
                         name="Outpatient", marker_color=BRIGHT))
    fig.add_hline(y=1.0, line_dash="dash", line_color=INK,
                  annotation_text="national median", annotation_position="top left")
    fig.update_yaxes(title="charge index (1.0 = national median)")
    fig.update_layout(barmode="group")
    return style(fig, 360)


# --- h008: outlier rate by chain ---
def fig_chains():
    df = t["chart_h008_chain"].sort_values("outlier_rate_pct")
    colors = [OK if v == 0 else FP if v >= 50 else MARIGOLD for v in df["outlier_rate_pct"]]
    fig = go.Figure(go.Bar(
        x=df["outlier_rate_pct"], y=df["chain"], orientation="h",
        marker_color=colors, text=[f"{v:.0f}%" for v in df["outlier_rate_pct"]],
        textposition="outside", cliponaxis=False))
    fig.add_vline(x=5.3, line_dash="dash", line_color=INK,
                  annotation_text="national 5.3%", annotation_position="top")
    fig.update_xaxes(title="% of chain's hospitals flagged as cost outliers")
    return style(fig, 340, legend=False)


# --- h013: archetype-adjusted steerage avoidances (flagged high-cost providers) ---
def fig_targets():
    df = t["chart_h013"].sort_values("adj_charge_idx")
    fig = go.Figure(go.Bar(
        x=df["adj_charge_idx"], y=df["hospital_label"], orientation="h",
        marker_color=FP, text=[f"{v:.1f}×" for v in df["adj_charge_idx"]],
        textposition="outside", cliponaxis=False))
    fig.add_vline(x=1.0, line_dash="dash", line_color=INK,
                  annotation_text="peer-archetype norm", annotation_position="top")
    fig.update_xaxes(title="charge index vs own archetype (1.0 = peer norm)")
    return style(fig, 380, legend=False)


# --- h009: triangulated driver importance ---
def fig_drivers():
    feats = ["state", "case-mix", "ownership", "n_drgs", "urbanicity", "archetype"]
    norm = lambda v: np.array(v) / np.sum(v)
    shap_ = norm([0.302, 0.251, 0.210, 0.085, 0.051, 0.066])
    perm_ = norm([0.279, 0.603, 0.247, 0.040, 0.017, 0.013])
    ebm_  = norm([0.202, 0.222, 0.145, 0.061, 0.054, 0.038])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=feats, y=shap_, name="SHAP", marker_color=DV_SAPPHIRE))
    fig.add_trace(go.Bar(x=feats, y=perm_, name="Permutation", marker_color=DV_PURPLE))
    fig.add_trace(go.Bar(x=feats, y=ebm_, name="EBM glass-box", marker_color=DV_PINK))
    fig.update_yaxes(title="share of importance")
    fig.update_layout(barmode="group")
    return style(fig, 360)


# --- h015: premium by archetype ---
def fig_premium_archetype():
    arch = ["Generalist", "Rehab-spec", "Academic", "Rural sm-acute", "Surgical-spec"]
    av = [64.6, 50.4, 46.7, 22.8, -11.0]
    pairs = sorted(zip(arch, av), key=lambda p: p[1])
    a, v = zip(*pairs)
    colors = [FP if x > 0 else CMP for x in v]
    fig = go.Figure(go.Bar(x=v, y=a, orientation="h", marker_color=colors,
                           text=[f"{x:+.0f}%" for x in v], textposition="outside",
                           cliponaxis=False))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title="for-profit premium (%)")
    return style(fig, 360, legend=False)


# --- dollar prize ---
def fig_prize():
    labels = ["For-profit<br>billed", "Excess<br>premium", "25% steer", "50% steer", "100% steer"]
    vals = [81.6, 25.8, 6.46, 12.92, 25.84]
    colors = [GRAY2, FP, OK, OK, OK]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors,
                           text=[f"${v:.1f}B" for v in vals], textposition="outside",
                           cliponaxis=False))
    fig.update_yaxes(title="$ billions (Medicare inpatient)")
    return style(fig, 360, legend=False)


# --- h012: service-mix similarity network (communities = archetypes), interactive ---
ARCHETYPES = sorted(t["net_nodes"]["archetype"].unique().tolist()) if "net_nodes" in t else []


def build_network_fig(arch="All archetypes", show_targets="on"):
    nodes = t["net_nodes"]
    edges = t["net_edges"]
    ex, ey = [], []
    for r in edges.itertuples(index=False):
        ex += [r.x0, r.x1, None]
        ey += [r.y0, r.y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=ex, y=ey, mode="lines", line=dict(width=0.4, color="rgba(61,60,56,0.09)"),
        hoverinfo="none", showlegend=False))

    if arch == "All archetypes":
        for grp in nodes["archetype_group"].value_counts().index:
            d = nodes[nodes["archetype_group"] == grp]
            fig.add_trace(go.Scattergl(
                x=d["x"], y=d["y"], mode="markers", name=grp,
                marker=dict(size=7, color=GROUP_COLORS.get(grp, "#999"),
                            line=dict(width=0.5, color="white")),
                text=d["archetype"], customdata=d["ccn"],
                hovertemplate="%{text}<br>CCN %{customdata}<extra>" + grp + "</extra>"))
    else:
        # focus mode: grey the rest, highlight the selected archetype
        oth = nodes[nodes["archetype"] != arch]
        sel = nodes[nodes["archetype"] == arch]
        fig.add_trace(go.Scattergl(
            x=oth["x"], y=oth["y"], mode="markers", name="other archetypes",
            marker=dict(size=5, color="rgba(150,148,140,0.30)"), hoverinfo="skip"))
        col = GROUP_COLORS.get(sel["archetype_group"].iloc[0], "#999") if not sel.empty else "#999"
        fig.add_trace(go.Scattergl(
            x=sel["x"], y=sel["y"], mode="markers", name=arch,
            marker=dict(size=9, color=col, line=dict(width=0.7, color="white")),
            text=sel["archetype"], customdata=sel["ccn"],
            hovertemplate="%{text}<br>CCN %{customdata}<extra></extra>"))

    if show_targets == "on":
        tg = nodes[nodes["is_target"]]
        if arch != "All archetypes":
            tg = tg[tg["archetype"] == arch]
    else:
        tg = nodes.iloc[0:0]
    if not tg.empty:
        fig.add_trace(go.Scatter(
            x=tg["x"], y=tg["y"], mode="markers+text", name="✕ Steerage avoidance (h013)",
            marker=dict(symbol="x", size=14,
                        color=[GROUP_COLORS.get(g, "#999") for g in tg["archetype_group"]],
                        line=dict(width=1.5, color="#3D3C38")),
            text=tg["label"], textposition="top center",
            textfont=dict(size=9, color="#3D3C38"),
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra>steerage avoidance</extra>",
            customdata=tg["archetype"]))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig = style(fig, 560, lock=False)  # network keeps zoom/pan — it's the explorable centerpiece
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, xanchor="left",
                                  yanchor="top", font=dict(size=11)),
                      margin=dict(l=6, r=10, t=10, b=6))
    return fig


def build_network_fig_static():
    """Static-export twin of build_network_fig: every (archetype-focus x avoidance-toggle)
    state lives in ONE figure as separate traces, each tagged with a `meta` dict the export's
    JS reads to flip visibility via Plotly.restyle — so the two dropdowns work in the browser
    with no Dash server. Initial visible state = All archetypes, avoidances on.
    meta.k: edge | all (all-archetypes base) | greybg (focus background) | focus | avoid."""
    nodes, edges = t["net_nodes"], t["net_edges"]
    ex, ey = [], []
    for r in edges.itertuples(index=False):
        ex += [r.x0, r.x1, None]
        ey += [r.y0, r.y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines",
        line=dict(width=0.4, color="rgba(61,60,56,0.09)"),
        hoverinfo="none", showlegend=False, meta={"k": "edge"}))
    # All-archetypes base — one trace per community group (visible initially)
    for grp in nodes["archetype_group"].value_counts().index:
        d = nodes[nodes["archetype_group"] == grp]
        fig.add_trace(go.Scattergl(x=d["x"], y=d["y"], mode="markers", name=grp,
            marker=dict(size=7, color=GROUP_COLORS.get(grp, "#999"), line=dict(width=0.5, color="white")),
            text=d["archetype"], customdata=d["ccn"],
            hovertemplate="%{text}<br>CCN %{customdata}<extra>" + grp + "</extra>",
            meta={"k": "all"}))
    # Focus background — all nodes greyed, reused for every focused archetype (hidden initially)
    fig.add_trace(go.Scattergl(x=nodes["x"], y=nodes["y"], mode="markers",
        marker=dict(size=5, color="rgba(150,148,140,0.30)"), hoverinfo="skip",
        showlegend=False, visible=False, meta={"k": "greybg"}))
    # Focus highlight — one per archetype (hidden initially)
    for a in ARCHETYPES:
        sel = nodes[nodes["archetype"] == a]
        col = GROUP_COLORS.get(sel["archetype_group"].iloc[0], "#999") if not sel.empty else "#999"
        fig.add_trace(go.Scattergl(x=sel["x"], y=sel["y"], mode="markers", name=a,
            marker=dict(size=9, color=col, line=dict(width=0.7, color="white")),
            text=sel["archetype"], customdata=sel["ccn"],
            hovertemplate="%{text}<br>CCN %{customdata}<extra></extra>",
            visible=False, meta={"k": "focus", "arch": a}))
    # Avoidance overlay — all (visible initially) + one per archetype (hidden)
    def _avoid(sub, arch_tag, vis):
        return go.Scatter(x=sub["x"], y=sub["y"], mode="markers+text",
            name="✕ Steerage avoidance (h013)",
            marker=dict(symbol="x", size=14,
                color=[GROUP_COLORS.get(g, "#999") for g in sub["archetype_group"]],
                line=dict(width=1.5, color="#3D3C38")),
            text=sub["label"], textposition="top center", textfont=dict(size=9, color="#3D3C38"),
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra>steerage avoidance</extra>",
            customdata=sub["archetype"], visible=vis, showlegend=True, meta={"k": "avoid", "arch": arch_tag})
    tg_all = nodes[nodes["is_target"]]
    fig.add_trace(_avoid(tg_all, "All archetypes", True))
    for a in ARCHETYPES:
        sub = tg_all[tg_all["archetype"] == a]
        if not sub.empty:
            fig.add_trace(_avoid(sub, a, False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig = style(fig, 560, lock=False)
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, xanchor="left",
                                  yanchor="top", font=dict(size=11)),
                      margin=dict(l=6, r=10, t=10, b=6))
    return fig


# --- steer TOWARD the cheapest same-archetype peer within 40 miles (ZIP-centroid distance) ---
# target, target_idx, peer, peer_idx, % cheaper, miles
STEER = [
    ("Capital Health Regional (NJ)", 6.79, "Jefferson Health-Northeast", 0.85, 87, 18),
    ("Carepoint Christ (NJ)", 6.02, "Carepoint Hoboken (same chain)", 5.12, 15, 2),
    ("Carepoint Bayonne (NJ)", 5.80, "Jamaica Hospital (NY)", 0.55, 91, 15),
    ("Capital Health Hopewell (NJ)", 4.85, "Thomas Jefferson Univ Hosp", 1.30, 73, 32),
    ("RMC San Jose (CA)", 4.82, "Stanford Valleycare", 2.16, 55, 31),
    ("North Houston Surgical (TX)", 4.23, "Tops Surgical Specialty", 0.97, 77, 6),
    ("Good Samaritan (CA)", 4.04, "Stanford Valleycare", 2.16, 46, 37),
    ("Stanford (CA)", 3.74, "Zuckerberg SF General", 2.29, 39, 27),
    ("Presby/St Luke's (CO)", 2.98, "Denver Health", 0.86, 71, 3),
]
# no comparable peer within 40 mi (network-adequacy gap) — nearest same-archetype cheaper is far
STEER_MISSES = [
    ("MLK Jr Community (CA)", "nearest comparable cheaper peer 345 mi away"),
    ("Carepoint Hoboken (NJ)", "nearest comparable cheaper peer 199 mi away"),
]


def fig_steer():
    rows = sorted(STEER, key=lambda r: r[1])  # ascending so highest at top
    lblx = 7.4  # all peer labels align in a column to the right of every dot
    fig = go.Figure()
    for tgt, ti, alt, ai, pct, mi in rows:
        fig.add_trace(go.Scatter(x=[ai, ti], y=[tgt, tgt], mode="lines",
                                 line=dict(color="#C9C6BE", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[r[3] for r in rows], y=[r[0] for r in rows], mode="markers",
        name="Steer toward (≤40 mi peer)", marker=dict(size=11, color=DV_TURQUOISE),
        text=[r[2] for r in rows],
        hovertemplate="%{text}<br>%{x:.1f}× norm<extra>cheaper peer ≤40 mi</extra>"))
    fig.add_trace(go.Scatter(
        x=[r[1] for r in rows], y=[r[0] for r in rows], mode="markers", name="Avoid (overpriced)",
        marker=dict(size=11, symbol="x", color=DV_TANGERINE), customdata=[r[1] for r in rows],
        hovertemplate="%{y}<br>%{customdata:.1f}× archetype norm<extra>avoid</extra>"))
    fig.add_trace(go.Scatter(
        x=[lblx] * len(rows), y=[r[0] for r in rows], mode="text",
        text=[f"{r[2]} · {int(r[5])} mi · −{int(r[4])}%" for r in rows],
        textposition="middle right", textfont=dict(size=9, color=INK),
        showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=1.0, line_dash="dash", line_color=INK,
                  annotation_text="norm", annotation_position="top")
    fig.update_xaxes(title="charge index vs own archetype (1.0 = peer norm)",
                     range=[0, 13.5], tickvals=[0, 2, 4, 6])
    return style(fig, 440)


# --- h011: rural exposure ---
def fig_rural():
    grp = ["Rural CAH", "Rural PPS", "Urban CAH", "Urban PPS"]
    mcr = [48.3, 27.6, 43.0, 24.3]
    neg = [67.5, 61.5, 53.3, 48.4]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp, y=mcr, name="Medicare day share %", marker_color=CMP))
    fig.add_trace(go.Bar(x=grp, y=neg, name="% with negative operating margin", marker_color=FP))
    fig.update_yaxes(title="percent")
    fig.update_layout(barmode="group")
    return style(fig, 360)


# ================================================================ LAYOUT HELPERS

def kpi(value, label, sub, cls=""):
    return html.Div(className="kpi", children=[
        html.Div(value, className=f"v {cls}"),
        html.Div(label, className="l"),
        html.Div(sub, className="s"),
    ])


def methods(*blocks):
    return html.Details(className="methods", children=[
        html.Summary("Methods & detail"),
        html.Div(className="methods-body", children=list(blocks)),
    ])


def mtable(header, rows):
    return html.Table([
        html.Thead(html.Tr([html.Th(h) for h in header])),
        html.Tbody([html.Tr([html.Td(c) for c in r]) for r in rows]),
    ])


def section(anchor, kicker, title, lede, body, takeaway, ok=False, methods_block=None):
    children = [
        html.Div(kicker, className="kicker"),
        html.H2(title),
        html.P(lede, className="lede"),
        body,
        html.Div(takeaway, className="takeaway ok" if ok else "takeaway"),
    ]
    if methods_block is not None:
        children.append(methods_block)
    return html.Section(id=anchor, className="section", children=children)


def chart(fig):
    return html.Div(className="card", children=[dcc.Graph(figure=fig, config={"displayModeBar": False})])


def chart2(f1, f2):
    return html.Div(className="grid2", children=[
        html.Div(className="card", children=[dcc.Graph(figure=f1, config={"displayModeBar": False})]),
        html.Div(className="card", children=[dcc.Graph(figure=f2, config={"displayModeBar": False})]),
    ])


# --- h018: cost x quality (CMS Hospital Compare star rating) ---
def fig_quality_scatter():
    d = t["chart_h018_scatter"]
    bg = d[~d["is_avoid"].astype(bool)]
    av = d[d["is_avoid"].astype(bool)]
    qcolor = lambda s: FP if s <= 2 else (MUTE if s == 3 else OK)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=bg["cost_idx"], y=bg["star_j"], mode="markers", name="all rated hospitals",
        marker=dict(size=5, color="rgba(150,148,140,0.28)"),
        customdata=bg["star"], hovertemplate="cost %{x:.2f}× · %{customdata}★<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=av["cost_idx"], y=av["star"], mode="markers+text", name="steerage avoidance (✕)",
        marker=dict(symbol="x", size=13, color=[qcolor(s) for s in av["star"]],
                    line=dict(width=1, color="#3D3C38")),
        text=av["short"], textposition="middle right", textfont=dict(size=8, color=INK),
        customdata=av["star"],
        hovertemplate="<b>%{text}</b><br>cost %{x:.2f}× · %{customdata}★<extra>avoidance</extra>"))
    fig.add_vline(x=1.2, line_dash="dash", line_color=INK,
                  annotation_text="above archetype norm →", annotation_position="top left")
    fig.add_hline(y=3.5, line_dash="dash", line_color=INK,
                  annotation_text="high quality (4–5★)", annotation_position="top right")
    fig.update_xaxes(type="log", title="cost index vs archetype (log; 1.0 = peer norm)",
                     tickvals=[0.5, 1, 2, 4, 6])
    fig.update_yaxes(title="CMS overall star rating", range=[0.4, 5.7], tickvals=[1, 2, 3, 4, 5])
    return style(fig, 460)


def fig_quality_bands():
    d = t["chart_h018_bands"].sort_values("ord")
    labels = d["band"].str.replace("\n", "<br>", regex=False)
    colors = [OK, OK, MUTE, FP]  # very-pricey tier flagged (lowest quality)
    fig = go.Figure(go.Bar(
        x=labels, y=d["avg_star"], marker_color=colors,
        text=[f"{v:.2f}★<br>{int(p)}% 4–5★" for v, p in zip(d["avg_star"], d["pct_4_5star"])],
        textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="avg CMS star rating", range=[0, 4])
    fig.update_xaxes(title="cost index band (n=2,616 rated)")
    return style(fig, 460, legend=False)


# ================================================================ APP

app = dash.Dash(__name__, title="UHC Affordability — hospital pricing")
server = app.server

NAV = [
    ("#s1", "Price spread"), ("#s2", "Markup"), ("#s3", "Where"),
    ("#snet", "Archetypes"), ("#s4", "Steerage"), ("#sq", "Quality"), ("#s5", "Drivers"), ("#s6", "Access"),
    ("#build", "How it's built"), ("#glossary", "Glossary"), ("#appendix", "Methods"),
]

app.layout = html.Div([
    # top bar
    html.Div(className="topbar", children=[
        html.Div([html.Span("UHC"), " Affordability · hospital pricing 2023"], className="brand"),
        html.Nav([html.A(lbl, href=href) for href, lbl in NAV]),
    ]),

    # hero
    html.Div(className="hero", children=[
        html.Div("CMS Medicare · inpatient + outpatient · 3,236 hospitals", className="eyebrow"),
        html.H1(["The same hospital care costs ", html.Em("up to 16× more"),
                 " depending on where you go — and that gap is largely pricing, not just cost of care."]),
        html.P("For a payer that reimburses a percentage of billed charges, the affordability "
               "lever is steerage: identical services carry wildly different submitted charges "
               "across hospitals, the spread is systematic by geography and ownership, and it "
               "shows up as markup behavior that disappears in formula-set Medicare payments. "
               "This dashboard walks the evidence from price spread to dollar opportunity.",
               className="lede"),
    ]),

    # KPI strip
    html.Div(className="kpis", children=[
        kpi("16×", "Same surgery, price range", "DRG 470 joint replacement: $20k → $316k across hospitals", "fp"),
        kpi("5.7×", "Median charge-to-Medicare markup", "Inpatient; 7.6× outpatient; 333 hospitals ≥10×", "fp"),
        kpi("+45%", "For-profit charge premium", "Net of case-mix, geography, archetype — but −9% on Medicare payment", "fp"),
        kpi("$12.9B", "Steerage prize (50%)", "Half the for-profit inpatient charge premium recovered", "ok"),
        kpi("ρ −0.03", "Cost ↔ quality", "Paying more buys no better care; the priciest tier is the lowest-rated (CMS stars)", "ok"),
    ]),

    html.Div(className="builtnote", children=[
        "Built with ", html.B("databench-mcp"), " — an AI-augmented analysis platform I designed and built for this "
        "role; all 18 hypotheses were driven and evidence-tracked through it. ",
        html.A("See how it was built →", href="#build"),
    ]),

    html.Div(className="divider"),

    # SECTION 1 — price spread
    section("s1", "01 · The problem",
            "Same service, very different price",
            "Submitted charges are extremely right-skewed (skew 12.5, top 1% of provider-DRG rows = "
            "16% of all billing), and for any single DRG the price varies several-fold across hospitals. "
            "Major joint replacement ranges 16× — $20k at the cheapest hospital to $316k at the most expensive.",
            chart2(fig_skew(), fig_dispersion()),
            ["Even for the identical procedure, ", html.B("where you go drives a 3–16× price swing"),
             ". Holding the DRG fixed, 94% of common DRGs still show a ≥3× spread from the 10th to 90th "
             "percentile hospital — that residual, hospital-level variation (not the procedure, which we hold "
             "constant here) is the raw material for steerage."],
            methods_block=methods(
                html.H4("Distribution & why medians/logs"),
                html.P(["Inpatient submitted charge is severely non-normal: ", html.B("skewness 12.5, kurtosis 535"),
                        ", KS normality p≈0. Mean $90,794 vs median $58,669; p95 $258,817, p99 $518,034, max $10.42M. "
                        "So the whole study uses medians, log transforms, and peer-relative (within-service / within-archetype) "
                        "comparison — never means/SD on raw dollars."]),
                html.H4("Concentration (Pareto heavy tail)"),
                html.P("Billing = charge × discharges. Top 1% of provider-DRG rows = 16.4% of total billed; "
                       "top 10% = 49.5%; top 20% = 65%; top 50% = 87.4%."),
                html.H4("Within-DRG dispersion"),
                html.P("Across 269 DRGs with ≥50 providers, median within-DRG p90/p10 charge ratio = 3.67×, median CV 0.58; "
                       "252/269 DRGs (94%) ≥3×; top: DRG 895 alcohol/drug rehab 9.4×, 073 cranial nerve 6.7×, 885 psychoses 5.3×."),
                html.H4("Worked example — DRG 470 (major joint replacement, 648 hospitals)"),
                mtable(["", "Charge"], [
                    ["National median", "$68,858 (p10 $38,673 · p90 $152,832)"],
                    ["Cheapest — Beverly Hospital, MA", "$19,972"],
                    ["Most expensive — Good Samaritan, CA", "$315,717"],
                    ["Range", "≈16× for the identical surgery"],
                ]),
                html.P([html.B("Tool:"), " ", html.Code("analyze_distribution"), " + ", html.Code("sql_query"),
                        " (concentration & within-DRG dispersion). Hypotheses h001, h002."]),
            )),

    html.Div(className="divider"),

    # SECTION 2 — markup
    section("s2", "02 · The mechanism",
            "Markup behavior, not higher cost",
            "Hospitals differ in how aggressively they mark up over what Medicare pays. The for-profit "
            "premium survives every control (case-mix, archetype, urbanicity, state) — and the falsification "
            "test settles the mechanism: the gap is +45% on submitted charges but −9% on formula-set Medicare payment.",
            html.Div([
                chart2(fig_premium_stages(), fig_falsification()),
                chart2(fig_markup(), fig_pricing_culture()),
            ]),
            ["The for-profit premium is ", html.B("charging behavior, not higher cost of care"),
             ". Real input costs (urban wages, cost of living) do lift the broader geographic spread — but "
             "Medicare's payment formula already wage-index-adjusts for them, so they can't explain the ownership "
             "gap: it runs +45% on charges yet −9% on that cost-adjusted Medicare payment. It only bites a "
             "charge-based commercial payer, and aggressive charging is a stable provider trait — hospitals "
             "expensive in inpatient are expensive in outpatient too (ρ = 0.82)."],
            methods_block=methods(
                html.H4("Markup definition & distribution"),
                html.P("Markup = median(submitted charge ÷ Medicare payment) per provider. Across 2,652 hospitals "
                       "(≥10 service lines): p10 3.40×, median 5.73×, p90 10.91×, p99 17.33×; 333 hospitals ≥10×. "
                       "By setting: inpatient median 5.52× (p90 10.81×), outpatient 7.62× (p90 16.06×) — outpatient markup is higher."),
                html.H4("Pricing culture transfers across settings"),
                html.P("Spearman ρ between each provider's case-mix-adjusted inpatient and outpatient charge index = "
                       "0.82 (2,847 hospitals active in both) — aggressive charging is a stable provider trait, not a per-service artifact."),
                html.H4("For-profit premium — OLS on log-charge, controls added stepwise"),
                mtable(["Controls", "For-profit premium"], [
                    ["raw", "+62.5%"], ["+ case-mix", "+55.2%"], ["+ archetype", "+55.9%"],
                    ["+ urbanicity", "+52.9%"], ["+ state (full)", "+46.3%"],
                ]),
                html.P(["Estimator is OLS on ", html.B("LN(charge)"), ", so each premium is ",
                        html.B("exp(β) − 1 = a geometric-mean ratio"), " — not least-squares on raw dollars, and "
                        "not quantile regression. The log transform is the skew remedy; for the right-skewed, "
                        "~log-normal charge data the geometric mean ≈ the median, so this stays consistent with the "
                        "medians/logs rule. Payment-side falsification uses the same estimator on LN(payment). "
                        "3-level OLS under full controls: nonprofit −33%, government −34% vs for-profit."]),
                html.H4("Falsification — pricing, not cost"),
                html.P(["Under identical full controls the for-profit gap is ",
                        html.B("+45% on submitted charge but −9% on formula-set Medicare payment"),
                        " — it's markup behavior, so it only bites a charge-based commercial payer, not Medicare."]),
                html.H4("\"But isn't it just cost of living?\""),
                html.P(["A fair objection: urban hospitals genuinely pay more for labor and real estate, so part of "
                        "the broader geographic price spread is real cost, not markup. That does not rescue a cost "
                        "explanation for the ownership premium, for two reasons. (1) The premium is measured ",
                        html.B("within state and within urbanicity"), ", so coarse regional cost differences are "
                        "already held constant. (2) The falsification's denominator — Medicare payment — is itself ",
                        html.B("adjusted by the CBSA-level area wage index"), " (IPPS/OPPS pay the labor share at "
                        "local wage rates), so cost-of-living is netted out. A genuine higher-cost hospital would "
                        "show ", html.B("higher"), " wage-adjusted Medicare payment; for-profits show −9%. Higher "
                        "input cost cannot move charges +45% while moving cost-adjusted payment in the opposite "
                        "direction — that signature is discretionary charge-setting, not cost."]),
                html.H4("Dollar sizing"),
                html.P("For-profit Medicare-inpatient billing ≈ $81.6B; excess premium ≈ $25.8B; recovering 50% via steerage ≈ $12.9B. "
                       "Illustrative, not a forecast: assumes charges proxy payer cost, the premium is fully attributable to markup, "
                       "and 50% of volume is steerable — the 25 / 50 / 100% scenario bars bound that last assumption."),
                html.P([html.B("Tools:"), " ", html.Code("sql_query"), ", ", html.Code("analyze_correlations"),
                        " (Spearman), nested OLS + LightGBM/SHAP (Python). Hypotheses h003, h014, h015."]),
            )),

    html.Div(className="divider"),

    # SECTION 3 — where it concentrates
    section("s3", "03 · The geography",
            "Where the cost concentrates",
            "Case-mix-adjusted charges span ~6× by state — Maryland (rate-regulated) at 0.35 to Nevada at 2.14. "
            "Metro hospitals charge 25–30% more than rural for the same service, while Medicare payment stays flat. "
            "Outlier rates cluster hard by chain: for-profit chains run 5–20× the independent rate; Kaiser and Ascension have zero.",
            html.Div([
                chart2(fig_states(), fig_urbanicity()),
                chart(fig_chains()),
            ]),
            ["High-leakage commercial markets are ", html.B("NV, CA, FL, TX, NJ"),
             "; Maryland's all-payer waiver is the regulated contrast. Cost concentrates by geography "
             "and for-profit chain ownership — not randomly."],
            methods_block=methods(
                html.H4("State index (case-mix adjusted)"),
                html.P("Each service-line charge ÷ national median for that exact DRG/APC, median per state (n_prov ≥ 5). "
                       "Range MD 0.352 → NV 2.144 (~6×). Mechanism via charge-vs-payment: Maryland charges lowest (0.35) yet "
                       "has the highest adjusted Medicare payment (1.557) — its all-payer global-budget waiver. Nevada charges "
                       "2.14× but Medicare pays only 1.08× = pure markup behavior."),
                html.H4("Urbanicity (case-mix adjusted)"),
                mtable(["Urbanicity", "IP charge idx", "OP charge idx"], [
                    ["Metro", "1.028", "1.040"], ["Micropolitan", "0.762", "0.817"],
                    ["Small town", "0.686", "0.747"], ["Rural", "0.713", "0.780"],
                ]),
                html.P("Medicare payment is ~flat across urbanicity, so the 25–31% metro premium is provider charging behavior. "
                       "Caveats: only 37 rural inpatient providers survive CMS cell-size suppression (survivorship bias); "
                       "Critical Access Hospitals are absent from MUP entirely."),
                html.H4("Outlier clustering by chain (within-archetype p95 watchlist, baseline 5.3%)"),
                mtable(["Group", "Outlier rate"], [
                    ["Capital Health / Carepoint / Tenet", "100%"], ["HCA (name-detected)", "24.4%"],
                    ["Independent", "4.5%"], ["Ascension / Kaiser (non-profit)", "0%"],
                ]),
                html.P("By state: CA 17.2%, NJ 14.5%, FL 13.5% vs PA 4.7%, NY 3.9%. Caveat: name-detection caught only ~90 of "
                       "~180 HCA hospitals (others land in 'Independent'), so the chain effect is a lower bound — which is exactly "
                       "what motivated pulling ownership from the HCRIS cost report."),
                html.P([html.B("Tool:"), " ", html.Code("sql_query"), " (case-mix-adjusted indices). Hypotheses h004, h005, h010, h008."]),
            )),

    html.Div(className="divider"),

    # SECTION 4 — service-mix network / archetypes (network analytics)
    section("snet", "04 · The structure",
            "Hospital archetypes — a service-mix similarity network",
            "Hospitals are nodes; an edge joins two hospitals whose service mix is similar (cosine "
            "similarity of their DRG/APC volume vectors, k-nearest-neighbour graph). Unsupervised "
            "community detection (Louvain) splits the 3,236-hospital network into 14 archetypes at "
            "modularity 0.70 — academic/tertiary centers, community generalists, surgical specialty "
            "shops, small rural/low-acuity, psychiatric, and rehab clusters. The communities are not "
            "hand-coded; they emerge from how hospitals actually deliver care. The ✕ marks flag the 11 "
            "steerage avoidances from the next section — note how they scatter across communities.",
            html.Div(className="card", children=[
                html.Div(className="net-controls", children=[
                    html.Div([html.Label("Focus archetype", className="ctl-label"),
                              dcc.Dropdown(id="net-arch", clearable=False, className="ctl-dd",
                                  value="All archetypes",
                                  options=[{"label": "All archetypes", "value": "All archetypes"}]
                                          + [{"label": a, "value": a} for a in ARCHETYPES])]),
                    html.Div([html.Label("Steerage avoidances (✕)", className="ctl-label"),
                              dcc.Dropdown(id="net-targets", clearable=False, className="ctl-dd",
                                  value="on",
                                  options=[{"label": "Shown", "value": "on"},
                                           {"label": "Hidden", "value": "off"}])]),
                ]),
                dcc.Graph(id="net-graph", figure=build_network_fig(),
                          config={"displayModeBar": False}),
            ]),
            ["The ✕ marks are the 11 archetype-adjusted steerage avoidances — and they ",
             html.B("sit in different communities, not one cluster"),
             " (generalist, small-rural, surgical, tertiary). That's the whole point: they're expensive "
             "relative to the hospitals doing the same kind of work, so comparing within-archetype — not "
             "to a national average — is what makes the flags defensible against “our patients are just sicker.”"],
            ok=True,
            methods_block=methods(
                html.H4("Construction"),
                html.P("Nodes = 3,236 hospitals; features = share of volume across 602 DRG/APC service "
                       "codes. Edges = cosine k-nearest-neighbours (k=10). Communities via Louvain "
                       "modularity maximization → 14 communities, modularity 0.704 (strong separation). "
                       "Archetypes named by signature-service lift (a community's share of a service ÷ the "
                       "national share). The graph shown is a stratified sample of ~800 hospitals laid out "
                       "with a seeded Fruchterman-Reingold force layout; colour = archetype group, hover = "
                       "specific archetype + CCN."),
                html.H4("The 14 archetypes"),
                mtable(["#", "Archetype", "Group"], [
                    ["0", "Community generalist — medical", "Generalist"],
                    ["1", "Community generalist — ortho/joint", "Generalist"],
                    ["2", "Community generalist — surgical (smaller)", "Generalist"],
                    ["3", "Small rural / low-acuity", "Rural-SmallAcute"],
                    ["4", "Academic / tertiary (cardiac-neuro)", "Academic-Tertiary"],
                    ["5", "Community — cardiac / stroke", "Generalist"],
                    ["6", "Rehab + specialty mix", "Rehab-Specialty"],
                    ["7", "Psychiatric / behavioral health", "Specialty-Behavioral"],
                    ["8", "Large surgical / cardiothoracic tertiary", "Academic-Tertiary"],
                    ["9", "Community generalist — medical/resp", "Generalist"],
                    ["10", "Surgical specialty — ortho/spine/endocrine", "Surgical-Specialty"],
                    ["11", "Rehab / rate-regulated (MD-heavy), low-markup", "Rehab-Regulated"],
                    ["12", "Ambulatory surgical / ophthalmology", "Surgical-Specialty"],
                    ["13", "Spine / neurosurgical specialty (highest charge)", "Surgical-Specialty"],
                ]),
                html.H4("Cross-checks & caveats"),
                html.P("PCA (variance spread across many components, no 2–3 dominant) and k-means "
                       "independently recover the coarse generalist-core / specialty-periphery structure, "
                       "so the communities aren't a Louvain artifact. Reproducibility caveat (logged as a "
                       "product gap): similarity_network's Louvain exposed no seed, so the partition was "
                       "exported and persisted to make it reproducible — the canonical labels reused here."),
                html.P([html.B("Tools:"), " ", html.Code("similarity_network"), " (Louvain) + ",
                        html.Code("run_model"), " (pca, kmeans) cross-check + ", html.Code("sql_query"),
                        " (signature-service lift). Hypothesis h012; feeds h013."]),
            )),

    html.Div(className="divider"),

    # SECTION 5 — steerage (avoidances + steer-toward peers)
    section("s4", "05 · The action",
            "Steer toward the cheaper in-market peer",
            "Flagging overpriced hospitals is only half the job — steerage means routing members TOWARD a "
            "lower-cost, comparable, in-network provider, and it's inherently local (a DC-metro member can't be "
            "sent to North Dakota). Left: the 11 hospitals expensive vs. their own archetype (the flag). Right: "
            "for each, the cheapest same-archetype peer within ~40 miles (great-circle distance from ZIP centroids — "
            "a drive-time proxy) to steer toward.",
            html.Div([
                chart2(fig_targets(), fig_steer()),
                html.P([html.B("No local option for 2 of 11: "),
                        "MLK Jr Community (nearest comparable cheaper peer 345 mi) and Carepoint Hoboken (199 mi) — "
                        "a network-adequacy gap, not a steerage win. (Robust: 99% of this archetype geocoded; see the "
                        "2013-ZIP-vintage caveat in the appendix.)"],
                       className="lede", style={"marginTop": "10px", "fontSize": "14px"}),
            ]),
            ["Steerage is ", html.B("toward the affordable comparable peer within ~40 miles"),
             ". 9 of 11 avoidances have one — Stanford → SF General (27 mi), not Ventura; Capital Health Regional → "
             "Jefferson Health-Northeast (18 mi, 87% cheaper). The other 2 have no comparable hospital within reach "
             "(199–345 mi) — a network-adequacy limit, and exactly why real routing needs drive-time + adequacy, not state. "
             "One guardrail this cost-only view omits: ", html.B("steer only where the cheaper peer is also high-quality"),
             " — cost is necessary, not sufficient (quality axis is in Future work)."],
            ok=True,
            methods_block=methods(
                html.H4("Two-layer method"),
                html.P("Layer 1: normalize each charge to the national median for that service (adj_charge_idx). "
                       "Layer 2: score within archetype on log(index) using a robust modified-z (median + MAD)."),
                html.H4("Flag counts (3,236 hospitals)"),
                mtable(["Method", "Flagged", "Note"], [
                    ["within-archetype p95", "153", "watchlist"],
                    ["robust modified-z ≥ 3.5", "11", "high-confidence (all also p95)"],
                    ["plain-z ≥ 2 (foil)", "83", "noisy"],
                    ["peer_outliers plain-z > 2", "15", "z inflated to 8.5 — SD dominated by outliers (masking made visible)"],
                ]),
                html.P("Robust method and the tool agree on the worst offenders, validating the avoidances while the robust cut is cleaner."),
                html.H4("Why archetype-adjusted (the defensibility argument)"),
                html.P("Raw within-service detection over-flags academic centers (Cedars-Sinai, Stanford, NYU) whose case-mix is "
                       "genuinely complex. Archetype-adjustment drops Cedars-Sinai (expensive like its academic peers) while "
                       "Stanford and the for-profit chains persist — so the list survives the 'but their patients are sicker' objection."),
                html.H4("The 11 high-confidence avoidances"),
                html.P("Capital Health Regional NJ (6.8×, markup 27×), Carepoint Christ / Bayonne / Hoboken NJ (6.0–5.1×), "
                       "Capital Health Hopewell NJ (4.9×, markup 28×), Regional Med Ctr San Jose CA (HCA), MLK Jr Community CA, "
                       "North Houston Surgical TX (physician-owned), Good Samaritan CA (HCA), Stanford CA, Presbyterian/St Luke's CO (HCA)."),
                html.H4("Direction & geography (steer toward, locally)"),
                html.P("Flagging uses national within-archetype comparison (defensible: like-for-like). But steerage is "
                       "directional and local — route members TOWARD a lower-cost comparable provider in their own market. "
                       "Picking the market scale matters, and I tested three:"),
                mtable(["Market definition", "Avoidances with a comparable cheaper peer"], [
                    ["Same state", "10 / 11 — but too loose (San Jose → Oroville ~3 hrs; Stanford → Ventura ~5 hrs)"],
                    ["Same ZIP3 region", "3 / 11 — too tight (nearby cheaper hospitals are a different archetype)"],
                    ["Within 40 mi (great-circle)", "9 / 11 — the right balance (used here)"],
                ]),
                html.P("The 40-mile market joins provider ZIP to public ZIP-centroid lat/long (ingested as "
                       "zip_centroids) and computes great-circle distance to same-archetype, lower-cost candidates. "
                       "Great-circle is a drive-time proxy (a few minutes' underestimate); production routing should use "
                       "true drive-time / Hospital Referral Region plus network-adequacy and service-level (not whole-"
                       "archetype) comparability. The 2 misses are real findings: MLK and Carepoint Hoboken have no "
                       "comparable cheaper hospital within reach (199–345 mi) — a network-adequacy gap. Implements "
                       "correction c007."),
                html.P([html.B("Data-quality caveat: "), "the ZIP→lat/long source is a 2013 vintage, so ZIPs added "
                        "since don't match — 93.6% of providers geocoded (~6% of candidate peers silently dropped). "
                        "All 11 avoidances (100%) and the small-rural archetype (99%) geocoded, so the findings hold, "
                        "but production needs a current ZIP/ZCTA vintage or address-level geocoding."]),
                html.P([html.B("Tools:"), " ", html.Code("similarity_network"), " (archetypes), ", html.Code("sql_query"),
                        " (robust-z + in-state peer join), ", html.Code("peer_outliers"), " (foil). Hypotheses h006, h007, h013."]),
            )),

    html.Div(className="divider"),

    # SECTION 5 — drivers + dollars
    section("sq", "06 · The quality gate",
            "Paying more doesn't buy better care",
            "Steerage on cost alone is unsafe — you could route a member to a cheap, low-quality hospital. So I "
            "joined CMS Hospital Compare (overall star rating) to the cost index. Two results: across 2,616 rated "
            "hospitals cost and quality are uncorrelated (Spearman −0.03), and the most expensive tier is actually "
            "the lowest-rated. And the avoidances split — most are expensive AND low-quality, but two are "
            "expensive-but-excellent and a cost-only list would wrongly punish them.",
            chart2(fig_quality_scatter(), fig_quality_bands()),
            ["The lever is ", html.B("low-cost AND high-quality"), " — 738 hospitals qualify. Most avoidances are "
             "1–2★ (steering away wins twice: cheaper and better), but the quality gate keeps ", html.B("Stanford "
             "(5★) and MLK Jr (4★)"), " off the steer-away list. Cost is necessary, not sufficient."],
            ok=True,
            methods_block=methods(
                html.H4("Source & join"),
                html.P(["CMS ", html.B("Hospital Compare — Hospital General Information"), " (overall star rating, "
                        "5,432 hospitals), joined on CCN to the archetype-adjusted cost index; 2,616 hospitals carry "
                        "both (smaller/specialty hospitals are unrated by CMS). Public CMS data — the takehome "
                        "explicitly invites supplementing with additional CMS datasets, so quality is in-scope."]),
                html.H4("Cost ≠ quality"),
                html.P("Spearman(cost index, star) = −0.03 (n=2,616). Mean star by cost band: 3.15 (cheap) → 3.26 → "
                       "3.22 → 2.74 (very pricey ≥2×); only 25% of the very-pricey tier is 4–5★ vs 39–44% elsewhere. "
                       "Higher price does not buy better outcomes — the priciest tier is the weakest."),
                html.H4("The 11 avoidances, quality-checked"),
                mtable(["Avoidance (cost vs archetype)", "Stars"], [
                    ["Carepoint Bayonne 5.8× / Hoboken 5.1× (NJ)", "1★ / 1★"],
                    ["Capital Health Regional 6.8× / Hopewell 4.9×, Carepoint Christ 6.0×, Good Samaritan 4.0×, Presby/St Luke's 3.0×", "2★"],
                    ["RMC San Jose 4.8× (CA)", "3★"],
                    ["MLK Jr Community 4.5× (CA) — safety-net, 'no local option' (h016)", "4★"],
                    ["Stanford 3.7× (CA) — premium partly buys quality", "5★"],
                    ["North Houston Surgical 4.2× (TX)", "n/a"],
                ]),
                html.P("7 of 10 rated avoidances are 1–2★ (steer-away doubly justified); the quality gate correctly "
                       "spares Stanford and MLK. Refines h013/h016."),
                html.P([html.B("Tools:"), " ", html.Code("ingest_file"), " (CMS Hospital Compare), ",
                        html.Code("sql_query"), " (CCN join, cost-band × star, Spearman). Hypothesis h018."]),
            )),

    html.Div(className="divider"),

    section("s5", "07 · The model",
            "What drives cost — and what it's worth",
            "A leakage-free provider-grain model (n=2,933) predicts cost moderately well — EBM R²≈0.70 (5-fold CV), "
            "≈$11k / 22% median dollar error. Three importance methods agree on the top trio: geography, case-mix, "
            "ownership. The for-profit premium concentrates in steerable generalist care, and sizing it puts the "
            "inpatient prize near $25.8B.",
            html.Div([
                chart2(fig_drivers(), fig_premium_archetype()),
                chart(fig_prize()),
            ]),
            ["Cost is predictable from a small factor set; ", html.B("geography and ownership are the levers, volume is not"),
             ". Recovering half the for-profit premium ≈ $12.9B on Medicare inpatient alone."],
            ok=True,
            methods_block=methods(
                html.H4("Leakage-free design"),
                html.P("Modeled at provider grain (one row per hospital, n=2,933) with a discharge-weighted case-mix index "
                       "(ln_expected_charge = national-median charge per DRG, weighted by the provider's discharge mix). This "
                       "preserves the DRG signal numerically while avoiding service-grain provider leakage (a hospital's own "
                       "pricing leaking across train/test) — the alternative being GroupKFold by provider."),
                html.H4("Models & metrics (5-fold CV, mean ± SD)"),
                mtable(["Model", "R² (log)", "MAE (log)", "$ median err", "$ median APE"], [
                    ["EBM (glass-box)", "0.704 ± .015", "0.270", "$11,090", "22%"],
                    ["LightGBM", "0.678 ± .029", "0.281", "$11,503", "22%"],
                    ["Lasso (LassoCV)", "0.584 ± .014", "0.324", "$13,592", "27%"],
                ]),
                html.P("Why these metrics, not R² alone: R² on a log target flatters fit and hides dollar error; MAE is "
                       "robust on heavy-tailed cost; the $ columns back-transform to what a stakeholder actually feels "
                       "(~$11k / ~22% typical miss). Reported as 5-fold CV mean ± SD — note EBM and LightGBM overlap "
                       "within ~1 SD, so the EBM edge is marginal, not decisive. (Lasso degenerates to R²≈0 at its default "
                       "α=1.0; LassoCV picks a working α. Adjusted R² isn't the fix — out-of-sample R² already penalizes "
                       "complexity.)"),
                html.H4("Triangulated importance (seed-stable)"),
                html.P("Interventional TreeSHAP (primary) + permutation + EBM glass-box agree on the top trio and are perfectly "
                       "rank-stable across 5 seeds (1-1, 2-2, 3-3): state > case-mix > ownership. Severity, volume, "
                       "specialization, archetype are minor."),
                html.H4("Variance decomposition (marginal η²)"),
                mtable(["Factor", "log charge", "log Medicare payment"], [
                    ["DRG (service)", "0.578", "0.869"], ["State", "0.180", "0.054"],
                    ["Urbanicity (RUCA)", "0.040", "0.011"], ["Volume", "0.001", "0.001"],
                ]),
                html.P("Charge is 58% service-determined vs payment 87%; geography's share triples (5.4%→18%). That ~30pp gap is "
                       "provider/geographic pricing discretion — the steerage lever. Volume is a non-factor (surprise vs the hypothesis)."),
                html.H4("Archetypes (h012)"),
                html.P("Cosine kNN service-mix network (3,236 hospitals × 602 codes, k=10) → 14 Louvain communities, modularity "
                       "0.704; signature-service lift names them (academic/tertiary, community generalist, small rural/low-acuity, "
                       "psychiatric, rehab/IRF, surgical-specialty). PCA + k-means cross-checks confirm the coarse structure."),
                html.P([html.B("Caveat:"), " all estimates are associational. A causal version (Double ML / causal forest with "
                        "market-concentration + wage-index confounders) is framed but not run — flagged as a databench-mcp product gap."]),
                html.P([html.B("Tools:"), " ", html.Code("run_model"), " (lasso/ebm/lightgbm/pca/kmeans), ",
                        html.Code("similarity_network"), ", SHAP/permutation (Python). Hypotheses h009, h012."]),
            )),

    html.Div(className="divider"),

    # SECTION 6 — access / rural
    section("s6", "08 · The constraint",
            "The access trade-off: rural exposure",
            "Steerage and reimbursement cuts land unevenly. Rural Critical Access Hospitals draw ~48% of inpatient "
            "days from Medicare (≈2× urban) yet two-thirds already run negative operating margins — they survive on "
            "cost-based reimbursement, not patient-care margin.",
            chart(fig_rural()),
            ["Rural hospitals combine the ", html.B("highest Medicare dependence with no margin cushion"),
             ". Affordability levers aimed at high-markup metro/for-profit hospitals avoid collateral damage to "
             "rural access — the two are largely different facilities."],
            methods_block=methods(
                html.H4("Source"),
                html.P("CMS HCRIS FY2023 Hospital Provider Cost Report (6,103 hospitals), ingested fresh because the MUP files "
                       "lack margins and payer mix. Grouped by CMS Rural/Urban × facility type (Critical Access vs PPS)."),
                html.H4("Medicare dependence & operating margin"),
                mtable(["Segment", "Medicare day share", "Median op. margin", "% negative op."], [
                    ["Rural CAH", "48.3%", "−6.17%", "67.5%"],
                    ["Rural PPS", "27.6%", "−5.32%", "61.5%"],
                    ["Urban PPS", "24.3%", "+0.11%", "48.4%"],
                ]),
                html.P("Rural CAHs draw ~half their inpatient days from Medicare (≈2× urban) yet two-thirds already lose money on "
                       "patient care — they survive on cost-based reimbursement, not operating margin. So a Medicare cut removes "
                       "the largest revenue source from facilities with zero cushion."),
                html.H4("Data-quality correction (logged)"),
                html.P("The first-pass total-margin metric (Net Income ÷ Total Income) was an artifact: CMS 'Total Income' already "
                       "nets operating expense and 'Total Other Expenses' is 72% null, forcing the ratio to ≈1.0. Corrected to "
                       "Net Income ÷ (Net Patient Revenue + Total Other Income) before recording — caught by adversarial review."),
                html.P([html.B("Tools:"), " ", html.Code("ingest_url"), " + ", html.Code("sql_query"), ". Hypothesis h011."]),
            )),

    html.Div(className="divider"),

    # SECTION 7 — how this was built (databench-mcp)
    html.Section(id="build", className="build", children=[html.Div(className="inner", children=[
        html.Div("09 · The tooling", className="kicker"),
        html.H2("How this was built — databench-mcp"),
        html.P(["This analysis wasn't run by hand. I designed and built ", html.B("databench-mcp"),
                " — an AI-augmented data-analysis MCP platform — "
                "purpose-built for this take-home and this AI-enablement role, then drove all 18 hypotheses through it. "
                "The platform is the methods story: it makes an LLM do rigorous, reproducible, auditable analysis instead "
                "of plausible-sounding guesses."], className="lede"),
        html.Div(className="buildgrid", children=[
            html.Div(className="bcard", children=[html.H4("Guard-railed tool surface"),
                html.P("MCP tools for ingestion, profiling, EDA, transforms, hypothesis tracking, statistics, modeling, "
                       "network analytics (cosine similarity graphs + Louvain community detection), visualization, and "
                       "reproducible recipes — each with built-in discipline (profile-before-model, natural-grain, no-leakage).")]),
            html.Div(className="bcard", children=[html.H4("Hypothesis-driven loop"),
                html.P(["ingest → profile → hypothesize → analyze → record evidence → repeat — a ",
                       html.B("CRISP-DM-style loop"), ". Evidence recording is mandatory "
                       "after every finding, enforced by the ", html.B("databench-analyst skill"), " (pinned in CLAUDE.md so it "
                       "survives context compaction) — so every claim on this page traces to a tool call and a logged note. "
                       "Result: 18/18 hypotheses supported, fully evidence-tracked."])]),
            html.Div(className="bcard", children=[html.H4("Human-in-the-loop correction ledger"),
                html.P("When the analyst corrects the AI's approach it's logged with a category. 13 corrections this project — "
                       "from foundational catches (target leakage; working out-of-order) to national-mean → archetype-adjusted "
                       "outliers, leakage-safe triangulated importance, steer-toward-not-away + geography-aware steerage, "
                       "R²-only → MAE + dollar-scale + CV metrics, the cost-of-living rebuttal, the 'steerage avoidance' "
                       "reframe, and the cost×quality gate — turning oversight into a durable, auditable record.")]),
            html.Div(className="bcard", children=[html.H4("Adversarial verification"),
                html.P("Findings default to 'suspect' until they survive refutation. This caught the AI early on using the "
                       "charge target as its own predictor (target leakage, falsely ranked #1) — forcing the leakage-free "
                       "provider-grain redesign — and later caught the HCRIS total-margin artifact.")]),
            html.Div(className="bcard", children=[html.H4("Self-aware product backlog"),
                html.P("Corrections flagged as product gaps become the tool's roadmap: seeds for stochastic methods, "
                       "auto-persisted cluster labels, and first-class causal-inference tooling (Double ML / causal forests) "
                       "to move from association to intervention.")]),
            html.Div(className="bcard", children=[html.H4("Reproducible & portable"),
                html.P("Every step is replayable as a recipe; this dashboard is generated from the same project artifacts. "
                       "The whole thing — analysis, platform, and this leave-behind — was built during the take-home window.")]),
        ]),
    ])]),

    html.Div(className="divider"),

    # APPENDIX
    html.Div(id="appendix", className="appendix", children=[
        html.H2("Appendix — methods, metrics & sources"),
        html.P("Everything below supports the story above; it's collapsed so the narrative stays clean. "
               "Expand any block for detail.", className="lede"),

        html.Details(id="glossary", className="app-block", open=True, children=[
            html.Summary("Glossary — terms & acronyms"),
            html.Div(className="app-body", children=[
                html.H4("Data & sources"),
                mtable(["Term", "Definition"], [
                    ["CMS", "Centers for Medicare & Medicaid Services — the federal agency; publisher of this data."],
                    ["MUP", "Medicare Provider Utilization & Payment Data — the public inpatient/outpatient charge files."],
                    ["HCRIS", "Healthcare Cost Report Information System — hospital cost reports (margins, ownership, payer mix)."],
                    ["CCN", "CMS Certification Number — the unique ID for each hospital."],
                    ["Submitted charge", "The hospital's list/billed price for a service (before any discount)."],
                    ["Medicare payment", "What Medicare actually pays — set by formula, not by the hospital."],
                ]),
                html.H4("Clinical grouping"),
                mtable(["Term", "Definition"], [
                    ["DRG", "Diagnosis-Related Group — inpatient classification of 'what was done' (e.g. DRG 470 = joint replacement)."],
                    ["APC", "Ambulatory Payment Classification — the outpatient equivalent of a DRG."],
                    ["Case-mix", "The mix and severity of services a hospital performs — sicker/complex vs routine."],
                    ["Severity tier", "How severe a DRG case is (affects expected cost)."],
                ]),
                html.H4("Hospital type & geography"),
                mtable(["Term", "Definition"], [
                    ["PPS", "Prospective Payment System hospital — paid by the standard Medicare formula."],
                    ["CAH", "Critical Access Hospital — small rural hospital reimbursed at cost (~101%), not by DRG."],
                    ["RUCA", "Rural-Urban Commuting Area codes — classify a location's urbanicity."],
                    ["Urbanicity", "Metro / micropolitan / small-town / rural classification."],
                    ["HRR", "Hospital Referral Region — a regional market for hospital care; the right scale for 'local' steerage."],
                    ["Type of Control", "Ownership: for-profit, nonprofit, or government (from HCRIS)."],
                ]),
                html.H4("Affordability concepts"),
                mtable(["Term", "Definition"], [
                    ["Markup", "Submitted charge ÷ Medicare payment — how aggressively a hospital prices over the formula rate."],
                    ["Charge index (adj_charge_idx)", "A provider's charge ÷ the national median for that exact service. 1.0 = national norm; 5× = five times the norm."],
                    ["Archetype", "A hospital's structural type, derived from its service mix (academic, generalist, surgical-specialty, …)."],
                    ["Steerage", "Routing members toward a lower-cost, comparable, in-network provider."],
                    ["Network adequacy", "Having enough in-network providers within reasonable reach of members."],
                    ["Leakage (affordability)", "Spend that escapes to high-cost providers — what a %-of-charges payer over-pays."],
                ]),
                html.H4("Statistics & modeling"),
                mtable(["Term", "Definition"], [
                    ["Median · p10 / p90 / p99", "The middle value; and the 10th / 90th / 99th percentiles (robust to outliers)."],
                    ["CV", "Coefficient of variation — spread relative to the average (SD ÷ mean)."],
                    ["Skewness / kurtosis", "Shape of a distribution — asymmetry / heaviness of the tail."],
                    ["Pareto / heavy tail", "A small share of items drives most of the total (e.g. top 1% = 16% of billing)."],
                    ["Modified-z / MAD", "A robust outlier score using the median + median-absolute-deviation (resists masking by outliers)."],
                    ["Spearman ρ", "Rank correlation between two variables (−1 to 1)."],
                    ["η² (eta-squared)", "Share of a variable's variance explained by one factor."],
                    ["R²", "How much of the outcome a model explains (0–1)."],
                    ["OLS / Lasso", "Linear regression / L1-regularized linear regression (drops weak features)."],
                    ["EBM", "Explainable Boosting Machine — an accurate 'glass-box' model you can read directly."],
                    ["LightGBM", "Gradient-boosted decision trees — high-accuracy 'black-box' model."],
                    ["SHAP / TreeSHAP", "A method that attributes a prediction to each feature's contribution."],
                    ["Permutation importance", "Feature importance measured by shuffling a feature and watching accuracy drop."],
                    ["GroupKFold", "Cross-validation that keeps each group (here, a provider) wholly in train or test — prevents leakage."],
                    ["Target leakage", "Accidentally using the answer (or a copy of it) as a predictor — gives a fake 'perfect' model."],
                    ["Associational vs causal", "A correlation/association is not proof of cause; causal claims need a causal design."],
                    ["Double ML / causal forest", "Methods that estimate cause-and-effect while controlling for confounders."],
                ]),
                html.H4("Network analytics"),
                mtable(["Term", "Definition"], [
                    ["Cosine similarity", "How alike two hospitals' service-mix vectors are by angle (0 = unrelated, 1 = identical)."],
                    ["kNN graph", "k-nearest-neighbours — connect each hospital to its k most similar peers."],
                    ["Louvain", "An algorithm that finds communities (clusters) in a network."],
                    ["Modularity", "Quality of a community split (0–1); 0.70 here means strong, well-separated communities."],
                    ["Community / module", "A densely-connected cluster of hospitals — here, a hospital archetype."],
                ]),
            ]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Human-in-the-loop correction ledger (databench-mcp)"),
            html.Div(className="app-body", children=[
                html.P("Every time the analyst corrected the AI's approach, databench-mcp logged it with a category — turning "
                       "oversight into an auditable record, and flagging gaps that became the tool's own roadmap. 9 logged this "
                       "project; the first two (◆) are foundational — they predate the formal ledger and are exactly what "
                       "prompted building the skill and guardrails:"),
                mtable(["#", "Category", "Correction"], [
                    ["c005 ◆", "data leakage ⚑gap", "Foundational: an early model (on Sonnet, before the skill) used submitted "
                     "charge — the target itself — as a predictor and ranked it the #1 cost driver. Classic target leakage. "
                     "Drop the target/derivatives; predict from exogenous drivers only. This forced the leakage-free "
                     "provider-grain redesign (case-mix index, not raw charge)."],
                    ["c006 ◆", "modeling discipline", "Foundational: the AI started ad hoc — outside databench-mcp and out of "
                     "order (toward modeling before profiling/EDA, no evidence tracking). Fixed by building the "
                     "databench-analyst skill that enforces the loop, now pinned in CLAUDE.md to survive compaction."],
                    ["c001", "domain methodology", "Step through one hypothesis at a time; add rurality & rural Medicare-exposure "
                     "questions; build a service-mix similarity network for archetype-adjusted (not national-mean) outliers; "
                     "supplement with HCRIS cost report; decide IP/OP combination per analysis."],
                    ["c002", "modeling discipline ⚑gap", "Stochastic methods (Louvain, k-means) need a seed; cluster labels "
                     "should auto-persist to a table instead of a manual JSON→CSV round-trip."],
                    ["c003", "reporting", "Miscounted open hypotheses after a session recovery (said 13/14, was 12/14) — read "
                     "statuses precisely before summarizing."],
                    ["c004", "domain methodology ⚑gap", "Move from single-model importance to triangulated TreeSHAP + permutation "
                     "+ EBM with provider-grouped CV; address that the whole analysis is associational — needs causal tooling "
                     "(Double ML / causal forests)."],
                    ["c007", "domain methodology", "Steerage is TOWARD a lower-cost comparable peer, not merely away from an "
                     "expensive one — name the affordable alternative. And it's local: a DC-metro member can't be routed to "
                     "North Dakota, so actionable steerage needs same-market (HRR / drive-time) + same-archetype + network "
                     "adequacy, not national comparison."],
                    ["c008", "statistical error ⚑gap", "Model comparison used a single split and R²-only on a log target. "
                     "Use MAE + back-transformed dollar error + k-fold CV mean±SD: R² hides dollar error (~$11k / 22% typical "
                     "miss) and the EBM-vs-LightGBM gap sits within fold noise. (Adjusted R² isn't the fix — out-of-sample "
                     "R² already penalizes complexity.) Raised as a question, logged as a correction."],
                    ["c009", "statistical error", "Section 02 chart fixes: discrete control sets were drawn as a line "
                     "(implies a trajectory) → now bars; and an OLS least-squares fit on right-skewed charge indices "
                     "(mislabeled with the Spearman ρ) → removed, replaced with log-log axes + an equal-pricing reference, "
                     "annotated as Spearman rank. Don't connect non-sequential categories, or conflate OLS with a rank correlation."],
                ]),
                html.P("◆ = foundational, pre-ledger (logged retroactively). ⚑gap = flagged as a databench-mcp product gap "
                       "(leakage guard, seeds, artifact persistence, causal inference) — the platform tracks its own backlog."),
            ]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Data sources"),
            html.Div(className="app-body", children=[
                html.P(["The assignment's core is the two CMS files — ", html.B("inpatient"), " and ",
                        html.B("outpatient"), " — and I used both, combined ", html.B("four ways depending on the "
                        "question"), ": (1) ", html.B("separately"), " at their natural grain (within-DRG inpatient + "
                        "within-APC outpatient) for outlier detection; (2) ", html.B("stacked long"), " (~210k rows) for "
                        "pooled cost-index and markup; (3) ", html.B("merged per-hospital"), " (both settings) for the "
                        "IP↔OP pricing-culture correlation (ρ=0.82) and the driver model; and (4) a ", html.B("combined "
                        "DRG+APC service-mix matrix"), " for the archetype network. The supplements below were added only "
                        "to answer the harder questions (ownership, rurality, geography, quality)."]),
                html.Ul([
                html.Li(["CMS Medicare Provider Utilization & Payment Data — ",
                         html.B("Inpatient 2023"), " (146,427 provider×DRG rows) and ",
                         html.B("Outpatient 2023"), " (116,799 provider×APC rows). ", html.B("The core datasets.")]),
                html.Li(["CMS ", html.B("HCRIS FY2023 Hospital Provider Cost Report"),
                         " (6,103 hospitals) — ownership (Type of Control), margins, Medicare day share. Ingested to answer h011/h015."]),
                html.Li("RUCA urbanicity codes joined to provider ZIP for rural/metro classification."),
                html.Li(["Public ", html.B("ZIP-centroid lat/long"), " (33k ZIPs, US gov 2013 — note: stale vintage, "
                         "93.6% provider match) — ingested to compute great-circle distance for the ~40-mile "
                         "'steer toward' market in Section 05."]),
                html.Li(["CMS ", html.B("Hospital Compare — Hospital General Information"), " (overall star rating, "
                         "5,432 hospitals) — joined on CCN for the §06 cost×quality gate (h018)."]),
            ])]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Assumptions & data-prep choices"),
            html.Div(className="app-body", children=[html.Ul([
                html.Li([html.B("Submitted charges as the affordability signal. "), "The load-bearing choice — charges "
                         "(chargemaster) are the lever a percent-of-billed payer feels. Allowed / negotiated rates aren't "
                         "in CMS public data (see Future work)."]),
                html.Li([html.B("Medians, logs, discharge-weighting. "), "Charge is skew-12.5, so every statistic uses "
                         "medians / log transforms / rank methods and volume-weights by discharges — never means or SD on "
                         "raw dollars."]),
                html.Li([html.B("Case-mix = charge ÷ national median for the exact DRG/APC. "), "Within-DRG severity is "
                         "approximated by an MS-DRG-description match (MCC/CC); refining with published DRG weights is Future work."]),
                html.Li([html.B("Ownership = HCRIS 'Type of Control' "), "collapsed to For-profit / Nonprofit / Government; "
                         "chain detection is name-based (undercounts HCA ~50%, so the for-profit effect is a lower bound)."]),
                html.Li([html.B("Archetypes = seeded Louvain "), "on a cosine service-mix kNN graph (stochastic; seed fixed "
                         "for reproducibility). 'In-market' = 40-mi great-circle from 2013 ZIP centroids. Markup = charge ÷ "
                         "Medicare payment as a pricing-behaviour proxy."]),
                html.Li([html.B("Exclusions. "), "RUCA=99 dropped; CMS cell-size suppression removes low-volume "
                         "provider-service rows (survivorship toward larger hospitals); Critical Access Hospitals are absent from MUP."]),
                html.Li([html.B("Quality = CMS overall star rating "), "joined on CCN; 2,616 of the cost-scored hospitals "
                         "are rated (smaller / specialty hospitals are unrated by CMS)."]),
            ])]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("What the databench-mcp tools do"),
            html.Div(className="app-body", children=[
                html.P("The analysis was driven entirely through these MCP tools — each tool chip in the section "
                       "methods above maps to one of these. Every tool carries built-in discipline "
                       "(profile-before-model, natural-grain, leakage-aware)."),
                mtable(["Tool", "What it does"], [
                    ["ingest_url / ingest_file", "Download or load a dataset into the project DB (raw saved + registered)."],
                    ["profile_table", "Types, null rates, cardinality, ranges — gates analysis (profile-before-model)."],
                    ["eda_summary", "One call: distributions + correlations + outlier overview for a table."],
                    ["analyze_distribution", "Distribution shape — skew, kurtosis, normality test, percentiles."],
                    ["analyze_correlations", "Correlation matrix + top pairs (Pearson or Spearman rank)."],
                    ["group_summary", "Group-wise statistics across a categorical variable."],
                    ["detect_outliers / peer_outliers", "Single-variable z-score outliers / entity-vs-peer (within-group) outliers."],
                    ["sql_query", "Read-only SQL against the project DuckDB (ad-hoc questions, joins)."],
                    ["run_model", "Fit Lasso / EBM / LightGBM / k-means / PCA with leakage-aware cross-validation."],
                    ["similarity_network", "Build a cosine-similarity k-NN graph and detect communities (Louvain)."],
                    ["hypothesis_add / record_evidence / update", "The hypothesis tracker — propose, attach evidence, set status."],
                    ["log_correction", "The human-in-the-loop correction ledger (categorized)."],
                    ["create_chart / build_dashboard", "Charts and an assembled dashboard from the analysis."],
                ]),
            ]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Method — why peer-relative, archetype-adjusted comparison"),
            html.Div(className="app-body", children=[
                html.P("Raw dollar means are meaningless on a skew-12.5 distribution, and a national-mean outlier "
                       "test over-flags academic centers whose case-mix is genuinely complex. The analysis uses: "
                       "(1) medians and log transforms throughout; (2) case-mix adjustment — each charge ÷ the national "
                       "median for that exact DRG/APC; (3) a cosine service-mix similarity network (Louvain, modularity "
                       "0.70) that partitions hospitals into 14 archetypes; (4) within-archetype robust scoring "
                       "(median + MAD modified-z ≥ 3.5) for the defensible steerage list."),
            ]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Model metrics — cost-driver triangulation"),
            html.Div(className="app-body", children=[
                html.P("Modeled at provider grain (one row per hospital, n=2,933) with a discharge-weighted "
                       "case-mix index so DRG signal is preserved without service-grain provider leakage."),
                html.Table([
                    html.Thead(html.Tr([html.Th(h) for h in
                        ["Model (5-fold CV)", "R² (log)", "MAE (log)", "$ median err", "$ median APE"]])),
                    html.Tbody([
                        html.Tr([html.Td(c) for c in ["EBM (glass-box)", "0.704 ± .015", "0.270", "$11,090", "22%"]]),
                        html.Tr([html.Td(c) for c in ["LightGBM", "0.678 ± .029", "0.281", "$11,503", "22%"]]),
                        html.Tr([html.Td(c) for c in ["Lasso (LassoCV)", "0.584 ± .014", "0.324", "$13,592", "27%"]]),
                    ]),
                ]),
                html.P("Metric choice (corr. c008): R² on a log target hides dollar error, so MAE + back-transformed "
                       "$ error are the decision-relevant numbers, reported as 5-fold CV mean ± SD. EBM and LightGBM "
                       "overlap within ~1 SD — the EBM edge is marginal. Separately, the η² variance decomposition "
                       "attributes 58% of log-charge variance to DRG (service) and 18% to geography."),
                html.P("Importance ranks are perfectly seed-stable across 5 seeds (state > case-mix > ownership). "
                       "Estimates are associational; a causal version (Double ML / causal forest with market-concentration "
                       "and wage-index confounders) is framed but not run."),
            ]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Caveats & data-quality notes"),
            html.Div(className="app-body", children=[html.Ul([
                html.Li("CMS volume suppression: only 37 rural inpatient providers survive the cell-size threshold — survivorship bias toward larger rural hospitals. Critical Access Hospitals are absent from MUP entirely (caught via HCRIS)."),
                html.Li("Chain name-detection caught ~90 of ~180 HCA hospitals; the rest ('Medical City', 'Good Samaritan', regional names) fall into 'Independent'. The for-profit/chain effect is therefore understated — a lower bound."),
                html.Li("HCRIS total-margin first pass was an artifact (CMS 'Total Income' already nets operating expense; 'Total Other Expenses' 72% null) — corrected to Net Income / (Net Patient Revenue + Other Income) before recording."),
                html.Li(["Steerage distance uses a ", html.B("2013-vintage ZIP→lat/long file"),
                         " — ZIPs created since don't match, so 93.6% of providers geocoded (187 dropped). The "
                         "≤40-mile peer search therefore silently omits ~6% of candidate peers. Checked: all 11 avoidances "
                         "and 99% of the small-rural archetype geocoded, so the 'no local option' finding holds — but a "
                         "current ZIP/ZCTA vintage (or address-level geocoding) is needed for production."]),
                html.Li("All effects are associational. Causal language is reserved for the framed-only DML design."),
            ])]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Temporal robustness — does it hold in 2022? (h017)"),
            html.Div(className="app-body", children=[
                html.P("Out-of-year holdout: re-ran the headline findings with identical SQL on the 2022 MUP files "
                       "(inpatient 145,742 rows; outpatient 116,459). Every one replicates — the patterns are structural "
                       "market features, not a 2023 artifact."),
                mtable(["Finding", "2022", "2023", "Holds?"], [
                    ["Inpatient charge skew", "11.4", "12.5", "✓"],
                    ["Within-DRG price dispersion (p90/p10)", "3.63×", "3.67×", "✓"],
                    ["Median markup — inpatient", "5.37×", "5.52×", "✓"],
                    ["Median markup — outpatient", "7.38×", "7.62×", "✓"],
                    ["Non-metro vs metro charge (case-mix adj.)", "−27.6%", "−26.6%", "✓"],
                    ["State cost-index ordering", "Spearman 0.99 vs 2023", "—", "✓"],
                    ["IP↔OP pricing-culture (Spearman)", "0.83", "0.82", "✓"],
                    ["For-profit charge premium (median, FP vs NP)", "+43%", "+49%", "✓"],
                ]),
                html.P("Most-expensive states are the same both years (NV, CA, AK, NJ, CO). Caveats: this validates the "
                       "patterns/effect sizes, not a hospital-for-hospital match (providers open/close); the for-profit "
                       "figure here is a median-ratio proxy (not the full nested OLS) and reuses sticky 2023 ownership. "
                       "Source: CMS MUP RY24 DY2022, re-encoded Latin-1 → UTF-8 (the 2023 files were pre-converted)."),
            ]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("Future work & expansions — what I'd build next"),
            html.Div(className="app-body", children=[
                html.P("Public-Medicare-data takehome — directional, not a production engine. The roadmap, in two tracks:"),
                html.H4("Analysis & data"),
                html.Ul([
                    html.Li([html.B("Care quality — extend the gate (priority deep-dive). "),
                             "§06 establishes cost and quality are separable (ρ −0.03). Next: ",
                             html.B("(a) drivers of quality — "), "regress the SAME factors that drive price "
                             "(geography, ownership, case-mix, archetype, volume) on quality as the outcome: do the "
                             "levers that raise price also move outcomes (e.g. does for-profit ownership predict lower "
                             "stars)? ", html.B("(b) a joint multivariate model — "), "price AND quality together "
                             "(seemingly-unrelated / multi-output regression, or a cost–quality efficiency frontier) so "
                             "steerage optimizes affordability and outcomes simultaneously, not as two separate axes. ",
                             html.B("(c) finer quality — "), "service-line mortality / readmission / complication by "
                             "condition, and quality-weighted savings."]),
                    html.Li([html.B("Prove cause, not just correlation. "), "Run the framed Double-ML / causal-forest "
                             "design (market-concentration HHI + wage-index confounders) to estimate the steerable effect."]),
                    html.Li([html.B("Network-aware steerage feasibility. "), "Layer in-network status, CMS time/distance "
                             "adequacy, true drive-time (not great-circle), service-line match (not whole-archetype), and "
                             "capacity — so a flagged peer is actually steerable."]),
                    html.Li([html.B("Real prices & commercial generalization. "), "Beyond CMS: Transparency-in-Coverage "
                             "MRFs (negotiated commercial rates) and plan claims, so the spread and dollar opportunity "
                             "reflect what we actually pay (chargemaster → allowed → negotiated) and are plan-specific."]),
                    html.Li([html.B("Access & equity guardrail. "), "Before down-tiering or steering away from a "
                             "community's only nearby hospital (the safety-net 'no local option' cases, e.g. MLK Jr), "
                             "check the access impact."]),
                    html.Li([html.B("Automate & refresh. "), "A re-runnable recipe with drift monitoring that rebuilds "
                             "each vintage; current ZIP/ZCTA + CCN/NPI chain crosswalk + Critical Access Hospitals."]),
                ]),
                html.H4("Strengthen the AI platform (enablement)"),
                html.Ul([
                    html.Li([html.B("Expand the model set + a method-selection skill "), "that picks the appropriate "
                             "method(s) for a given analysis (nested CV, broader families, calibration)."]),
                    html.Li([html.B("Prevent repeat mistakes. "), "Turn each logged correction into an automated "
                             "guardrail (a check that fires before the same error can recur), tracked as git issues — "
                             "not just a one-off fix."]),
                    html.Li([html.B("Privacy & compliance guardrails "), "(HIPAA / PHI, GDPR) built into the tool surface."]),
                    html.Li([html.B("Internal knowledge base (wiki). "), "Persist methods, decisions, hypotheses and the "
                             "correction ledger as a searchable team resource — institutional memory."]),
                ]),
            ]),
        ]),

        html.Details(className="app-block", children=[
            html.Summary("All 18 hypotheses — status"),
            html.Div(className="app-body", children=[
                html.Table([
                    html.Thead(html.Tr([html.Th("ID"), html.Th("Hypothesis"), html.Th("Status")])),
                    html.Tbody([html.Tr([html.Td(h[0]), html.Td(h[1]),
                                         html.Td(html.Span("supported", className="pill sup"))]) for h in [
                        ("h001", "Submitted charges extremely right-skewed (heavy tail)"),
                        ("h002", "Same DRG, wide cross-provider price dispersion — the steerage opportunity"),
                        ("h003", "Charge-to-payment markup varies widely; a subset marks up far above peers"),
                        ("h004", "Cost per service differs systematically by state/region"),
                        ("h005", "Rurality matters for cost per service (folded into h010)"),
                        ("h006", "Within-DRG inpatient high-cost outlier providers"),
                        ("h007", "Within-APC outpatient high-cost outlier providers"),
                        ("h008", "Outliers cluster by state and for-profit chain (repeat offenders)"),
                        ("h009", "Cost is predictable from a small factor set; service type dominant"),
                        ("h010", "Non-metro hospitals charge 25–30% less for identical service"),
                        ("h011", "Rural hospitals more exposed to Medicare-funding loss (thin margins, high dependence)"),
                        ("h012", "Service-mix network partitions into real hospital archetypes"),
                        ("h013", "Archetype-adjusted outliers = defensible steerage avoidances"),
                        ("h014", "Pricing culture transfers across IP/OP settings (ρ=0.82)"),
                        ("h015", "For-profit ownership → higher charges, as markup not cost (+45% / −9%)"),
                        ("h016", "Steerage is geographically bounded — 2 of 11 avoidances have no comparable peer within 40 mi"),
                        ("h017", "Temporal robustness — all headline findings replicate in 2022 (out-of-year holdout)"),
                        ("h018", "Cost and quality uncorrelated (ρ −0.03) — steerage must be cost × quality (CMS Hospital Compare)"),
                    ]]),
                ]),
            ]),
        ]),
    ]),

    html.Footer(className="foot", children=[
        "Generated by databench-mcp · project uhc_affordability · 2023 CMS Medicare data · "
        "18/18 hypotheses supported · all effects associational."
    ]),
])

@app.callback(
    Output("net-graph", "figure"),
    Input("net-arch", "value"),
    Input("net-targets", "value"),
)
def _update_network(arch, show_targets):
    return build_network_fig(arch or "All archetypes", show_targets or "on")


if __name__ == "__main__":
    app.run(debug=False)
