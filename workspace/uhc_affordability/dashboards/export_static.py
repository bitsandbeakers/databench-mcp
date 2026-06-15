"""
Static export of the Dash app -> a single self-contained index.html for hosting on a
static site (e.g. the CV / Cloudflare chain). No Python server needed.

Single source of truth: this walks app.app.layout (the SAME layout the live app uses) and
serializes Dash components to HTML, so there's no duplicated content to maintain. Plotly
figures stay interactive (hover, zoom, legend-click to toggle series). The two network
dropdowns become a disabled control + legend-click (the server callback can't run statically).

Run:  uv run --with pyarrow python export_static.py
Output: export/index.html
"""
import html as _html
from pathlib import Path
import plotly.io as pio
from plotly.offline import get_plotlyjs
import app  # reuse the live layout + figures

HERE = Path(__file__).parent
OUT = HERE / "export"
OUT.mkdir(exist_ok=True)

VOID = {"img", "br", "hr", "meta", "input"}
# camelCase style keys -> kebab-case
def _kebab(k):
    return "".join("-" + c.lower() if c.isupper() else c for c in k)

def _style(d):
    parts = []
    for k, v in d.items():
        if isinstance(v, (int, float)) and k not in ("opacity", "zIndex", "flex", "lineHeight", "fontWeight"):
            v = f"{v}px"
        parts.append(f"{_kebab(k)}:{v}")
    return ";".join(parts)

def _attrs(props):
    a = []
    if props.get("className"):
        a.append(f'class="{_html.escape(props["className"])}"')
    if props.get("id"):
        a.append(f'id="{_html.escape(props["id"])}"')
    if props.get("href"):
        a.append(f'href="{_html.escape(props["href"])}"')
    if props.get("style"):
        a.append(f'style="{_html.escape(_style(props["style"]))}"')
    if props.get("open"):
        a.append("open")
    return (" " + " ".join(a)) if a else ""

def render(node):
    if node is None:
        return ""
    if isinstance(node, (str, int, float)):
        return _html.escape(str(node))
    if isinstance(node, list):
        return "".join(render(n) for n in node)
    # Dash component
    obj = node.to_plotly_json() if hasattr(node, "to_plotly_json") else node
    typ = obj.get("type", "")
    props = obj.get("props", {})

    if typ == "Graph":
        # topojsonURL -> sibling file so the choropleth renders without the CDN (self-contained)
        cfg = {"displayModeBar": False, "topojsonURL": "./"}
        if props.get("id") == "net-graph":
            # client-side interactive network: all states baked into one figure, switched by
            # the dropdowns via Plotly.restyle (see NET_JS) — no Dash server needed.
            return pio.to_html(app.build_network_fig_static(), full_html=False,
                               include_plotlyjs=False, config=cfg,
                               default_width="100%", div_id="net-graph-div")
        return pio.to_html(props.get("figure"), full_html=False, include_plotlyjs=False,
                           config=cfg, default_width="100%")
    if typ == "Dropdown":
        opts = props.get("options", [])
        val = props.get("value")
        did = props.get("id", "")
        # enabled + carries its id and option VALUES so NET_JS can read select.value
        sel = "".join(
            f'<option value="{_html.escape(str(o.get("value","")))}"'
            f'{" selected" if o.get("value")==val else ""}>{_html.escape(str(o.get("label","")))}</option>'
            for o in opts)
        return f'<select class="ctl-dd" id="{_html.escape(did)}">{sel}</select>'

    tag = typ.lower()
    inner = render(props.get("children"))
    if tag in VOID:
        return f"<{tag}{_attrs(props)}>"
    return f"<{tag}{_attrs(props)}>{inner}</{tag}>"

css = (HERE / "assets" / "style.css").read_text(encoding="utf-8")
plotlyjs = get_plotlyjs()  # inline so the page is fully self-contained (no CDN dependency)
body = render(app.app.layout)

# Client-side replacement for the Dash network callback: read the two <select>s and flip
# trace visibility on the combined figure (each trace tagged with meta.k / meta.arch).
NET_JS = """<script>
(function(){
  function wire(){
    var gd=document.getElementById('net-graph-div');
    var a=document.getElementById('net-arch'), t=document.getElementById('net-targets');
    if(!gd||!a||!t||!gd.data){return setTimeout(wire,150);}
    function apply(){
      var arch=a.value, tog=t.value;
      var vis=gd.data.map(function(tr){
        var m=tr.meta||{};
        if(m.k==='edge')return true;
        if(m.k==='all')return arch==='All archetypes';
        if(m.k==='greybg')return arch!=='All archetypes';
        if(m.k==='focus')return arch===m.arch;
        if(m.k==='avoid')return arch===m.arch && tog==='on';
        return true;
      });
      Plotly.restyle(gd,{visible:vis});
    }
    a.addEventListener('change',apply);
    t.addEventListener('change',apply);
    apply();
  }
  wire();
})();
</script>"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UHC Affordability — hospital pricing 2023</title>
<script charset="utf-8">PLOTLY_JS_PLACEHOLDER</script>
<style>{css}
/* static-export note */
.static-note{{font-size:12px;color:var(--muted);margin:0 0 8px;}}
select.ctl-dd{{padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);min-width:240px;}}
</style>
</head>
<body>
<div class="static-note" style="max-width:1180px;margin:0 auto;padding:8px 32px 0;">
Static export — fully self-contained. Charts show details on hover and legend-click toggles series;
the network's archetype focus and avoidance toggle work right here in the browser. Zoom/pan is
enabled only on the network graph.
</div>
{body}
NETWORK_JS_PLACEHOLDER
</body>
</html>
"""
doc = doc.replace("PLOTLY_JS_PLACEHOLDER", plotlyjs)  # after f-string build — JS has braces
doc = doc.replace("NETWORK_JS_PLACEHOLDER", NET_JS)
(OUT / "index.html").write_text(doc, encoding="utf-8")
kb = len((OUT / "index.html").read_bytes()) // 1024
cdn = "cdn.plot.ly" in doc
print(f"wrote export/index.html ({kb} KB) · plotly figures: {doc.count('plotly-graph-div')} · CDN ref: {cdn}")
