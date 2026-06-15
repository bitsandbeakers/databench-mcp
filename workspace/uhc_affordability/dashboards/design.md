# Design system — UHC Affordability dashboard (Optum brand)

Design scheme for the dashboard, derived from the Optum brand system. This file is the
single source of truth for color, type, and layout; `assets/style.css` and the figure
palette in `app.py` implement it.

## Provenance

Color, typography, and data-viz values are read directly from the official Optum Brand
Center (`/content/color`, `/content/typography`, `/content/data-visualization`) — these are
authoritative. Every value is tagged:

- **[official]** — read from the Optum Brand Center.
- **[derived]** — chosen by us to fill a role Optum documents but whose exact hex we have
  not yet read; selected to read as Optum and meet WCAG 2.1 AA. Replace on capture.

---

## 1. Color  *(read from brand.optum.com/content/color)*

### Brand palette [official]
| Token | Hex | Pantone / RGB | Role (per brand site) |
|-------|-----|---------------|------------------------|
| `--optum-orange` | `#FF612B` | PMS 165 · 255,97,43 | **The star.** Hero — full floods to mini pops. Chrome, kickers, KPI accents. *Never as text (fails contrast).* |
| `--warm-white` | `#FAF8F2` | 250,248,242 | Primary background |
| `--warm-gray` | `#3D3C38` | PMS 2336C · 61,60,56 | **Text color** (11.04:1 on white) |
| `--sky-blue` | `#D9F6FA` | 217,246,250 | Accent color — *not for backgrounds* |
| `--white` | `#FFFFFF` | 255,255,255 | Background / card surface |
| `--black` | `#000000` | 0,0,0 | Print only; do not use black type on screen |

### Special-use palette [official]
| Token | Hex | RGB | Rule |
|-------|-----|-----|------|
| `--dawn-orange` | `#FFD1AB` | 255,209,171 | Broadens orange range; light tint backdrops |
| `--sunset-orange` | `#F9A667` | 249,166,103 | Lifts Optum Orange; only with Sky Blue present |
| `--dark-orange` | `#D74120` | 215,65,32 | **Excess / problem emphasis.** Tone-on-tone with Optum Orange only |
| `--horizon` | `#EDE8E0` | 237,232,224 | Tone-on-tone with Warm White only |

### Supporting palettes [official]
| Token | Hex | RGB | Role |
|-------|-----|-----|------|
| `--hyperlink` | `#095F87` | 9,95,135 | Links / interactive text (primary 70). Digital only. |

> **Data-viz & alert colors (success green, error red, categorical series) live on a
> separate page — `/content/data-visualization` — still to be read.** Until then the chart
> green/red below are [derived].

### Brand chrome (non-data) — Optum Orange as a *pop*, never as type
Optum Orange appears as **fills/shapes** (topbar accent, section heading bar, KPI card
top-accent), never as colored body text (fails contrast; brand rule). All text is Warm Gray.

### Accessibility (verified on brand site)
Warm Gray on White 11.04:1 · on Warm White 10.4:1 · on Sky Blue 9.73:1. **Do not** use
Optum Orange as text (3.1:1 on white, 2.82:1 on warm white — both fail AA). No black type;
no tints of brand colors; no grayscale conversion of brand colors.

### Neutrals
| Token | Hex | Source | Role |
|-------|-----|--------|------|
| `--ink` | `#3D3C38` | [official] Warm Gray | Primary text |
| `--muted` | `#3D3C38` @ 70% | [derived] | Muted text, axis labels |
| `--line` | `#EDE8E0` | [official] Horizon | Borders, dividers |
| `--bg` | `#FAF8F2` | [official] Warm White | Page background |
| `--card` | `#FFFFFF` | [official] | Card surface |

---

## 2. Typography  *(read from brand.optum.com/content/typography)*

Two primary typefaces [official]:
- **Enterprise Sans** — approachable, clean, modern. **Body copy** (Regular) and
  **headlines/subheads** (Bold). The workhorse for digital/tech audiences.
- **Enterprise Serif** (Enterprise Serif Text) — human, trustworthy. **Large display
  headlines and emphasis only — never body copy.**
- **Variable fonts** (Enterprise Sans/Serif Variable) are specified for "almost all
  digital spaces" — websites and apps. This dashboard targets those.
- **Fallback** [official]: Georgia Pro / **Arial Bold** for headlines, **Arial Regular**
  for body when brand fonts are unavailable (our case — fonts not bundled).

CSS stacks used:
- Sans: `"Enterprise Sans", "Enterprise Sans Variable", Arial, system-ui, sans-serif`
- Serif (emphasis/display): `"Enterprise Serif Text", "Enterprise Serif", "Georgia Pro", Georgia, serif`

**Emphasis is typographic, not color** [official rule]. Mix Serif into Sans for emphasis
(e.g. hero key phrase in Enterprise Serif italic/regular). Max 2 weights/styles per
message, ≥1 weight apart, once per page.

- **Scale** (screen, no official px scale published — kept readable):
  Hero H1 40px · Section H2 26px · KPI value 30px · body 16px · small 13px · eyebrow 12px.
- **Type color:** Warm Gray `#3D3C38` (10.4–11:1). Min contrast 4.5:1, AA at 11pt.

**Watchouts** [official]: no Optum Orange type; no dark-blue type; no Serif body copy; no
italics as a main font; don't highlight words with color; no title case (except proper
names); no all-caps unless legal; ≤2 styles for emphasis.

---

## 3. Data visualization  *(read from brand.optum.com/content/data-visualization)*

Optum keeps a **separate, theme-agnostic data palette** so charts don't compete with brand
moments. **Use this palette for charts — not the brand orange.** Color is never the only
cue: pair with labels, legends, shapes, direction. Don't mix categorical / sequential /
diverging in one chart.

### Categorical — official sequence (apply strictly in order) [official]
| # | Name | Hex | RGB |
|---|------|-----|-----|
| 1 | Turquoise 60 | `#15A796` | 21,167,150 |
| 2 | Pink 60 | `#C72887` | 199,40,135 |
| 3 | Purple 60 | `#8061BC` | 128,97,188 |
| 4 | Tangerine 60 | `#E4780C` | 228,120,12 |
| 5 | Sapphire 60 | `#1E82CB` | 30,130,203 |

Each primary has dark/light/background + child shades on Design Standards (SSO-gated, not
captured). Build extra shades as tints of these where needed [derived tints].

### Mapping to this dashboard's data roles
| Data role | Token | Basis |
|-----------|-------|-------|
| For-profit / excess / problem | Tangerine 60 `#E4780C` | [official] warm = high/attention |
| Comparison / regulated | Sapphire 60 `#1E82CB` | [official] |
| Opportunity / steerable | Turquoise 60 `#15A796` | [official] reads positive |
| Secondary series (e.g. outpatient) | Pink 60 `#C72887` | [official] |
| Third / method series | Purple 60 `#8061BC` | [official] |
| Neutral baseline | Warm Gray `#3D3C38` | [official] |

### Sequential (choropleth / intensity)
Single-hue lightness ramp on **Tangerine** (cost intensity): `#FBE7CE`, `#F4B26B`,
`#E4780C`, `#A6560A` [derived tints of official Tangerine 60 — exact child shades SSO-gated].

### Alert states (success / error / warning)
Hex values are on Design Standards (SSO-gated) — **not captured**. The dashboard uses
Turquoise for the positive/"opportunity" role, which covers the only green need; no
error/warning state is currently rendered. Capture if alerts get added.

---

## 4. Shape, spacing & elevation

- **Corner radius:** cards 14px, KPI 14px, pills 999px, inputs 10px. Optum digital is
  soft-but-not-pill; 12–16px on containers.
- **Spacing:** 8px base grid. Section padding 44px vertical / 32px horizontal; card padding
  ~20px; KPI gap 18px.
- **Elevation:** flat with hairline borders (`--gray-light`) and a single soft shadow
  `0 1px 2px rgba(20,30,50,.04)`. No heavy drop shadows — clean, clinical, airy.
- **Max content width:** 1180px, centered.

---

## 5. Components

All text is **Warm Gray**. Optum Orange appears only as a **shape/fill pop**.

- **Topbar:** Warm-White 92% + blur, Horizon hairline border; brand mark = small Optum
  Orange square + "UHC Affordability" in Warm Gray; nav links Warm Gray, hover `--hyperlink`.
- **Hero:** large H1 in Warm Gray; key phrase emphasized in **Enterprise Serif italic**
  (typographic emphasis, not color).
- **KPI cards:** white surface, Horizon border, **Optum Orange top-accent bar** (the brand
  pop); value colored by data role (Tangerine = problem, Turquoise = opportunity).
- **Section kicker:** uppercase Warm Gray with a short **Optum Orange leading bar**.
- **Takeaway callout:** left border + faint tint by sentiment — problem = Tangerine on
  `#FDF1E7`; opportunity = Turquoise on `#EAF6F4`.
- **Appendix:** `<details>` blocks, Horizon border, `+`/`−` affordance; supported pill on
  faint Turquoise.
- **Charts:** transparent background, Horizon gridlines, Warm-Gray reference lines, legend
  top-left; series from the data-viz palette in sequence.

---

## 6. Accessibility

- Text is Warm Gray on White/Warm-White (10.4–11:1) — well past AA.
- Color is never the only cue (brand data-viz rule): pair with labels, legends, direction.
- Categorical series use the official high-contrast sequence (Turquoise→Pink→Purple→
  Tangerine→Sapphire), separated by hue; safe across common color-vision deficiencies.
- Optum Orange is never used as text (3.1:1 fails); links use `--hyperlink #095F87`.
