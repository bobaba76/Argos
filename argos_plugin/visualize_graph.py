#!/usr/bin/env python3
"""Generate an interactive HTML graph visualization from Argos's Kuzu database.

Usage:
    python visualize_graph.py                    # generates graph_viz.html (default 200 nodes)
    python visualize_graph.py --limit 1000       # more nodes (slower but tolerable without physics)
    python visualize_graph.py --limit all        # all nodes (big file, still usable)

Opens browser when done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path


def _get_hermes_home() -> Path:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "hermes-agent"))
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        pass
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    local = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))
    if local.exists():
        return local
    return Path(os.path.expanduser("~/.hermes"))


def load_graph(home: Path, limit: int) -> tuple[list[dict], list[dict], int, int]:
    graph_path = home / "hybrid_memory_kuzu"
    if not graph_path.exists():
        print(f"  [ERROR] Graph database not found at {graph_path}")
        sys.exit(1)

    try:
        import kuzu
    except ImportError:
        print("  [ERROR] kuzu not installed.")
        sys.exit(1)

    try:
        db = kuzu.Database(str(graph_path))
        conn = kuzu.Connection(db)
    except Exception as e:
        print(f"  [ERROR] Cannot open graph (locked by memory service?): {e}")
        sys.exit(1)

    # Get total counts
    result = conn.execute("MATCH (n:Entity) RETURN COUNT(*)")
    total_nodes = result.get_next()[0]
    result = conn.execute("MATCH ()-[r:RelatesTo]->() RETURN COUNT(*)")
    total_edges = result.get_next()[0]

    nodes = []
    node_map = {}

    result = conn.execute("MATCH (n:Entity) RETURN n.id, n.entity_type, n.user_scope LIMIT $1", {"1": limit})
    while result.has_next():
        row = result.get_next()
        node_id = str(row[0])
        node_type = str(row[1] or "unknown")
        scope = str(row[2] or "global")
        nodes.append({"id": node_id, "type": node_type, "scope": scope})
        node_map[node_id] = True

    edges = []
    result = conn.execute(
        "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) "
        "RETURN a.id, r.relation_type, b.id"
    )
    while result.has_next():
        row = result.get_next()
        src = str(row[0])
        rel = str(row[1] or "relates_to")
        tgt = str(row[2])
        if src in node_map and tgt in node_map:
            edges.append({"source": src, "relation": rel, "target": tgt})

    conn.close()
    return nodes, edges, total_nodes, total_edges


def generate_html(nodes: list[dict], edges: list[dict], total_nodes: int, total_edges: int, output_path: str) -> str:
    type_colors = {
        "person": "#4CAF50", "concept": "#2196F3", "place": "#FF9800",
        "organization": "#9C27B0", "event": "#F44336", "role": "#00BCD4",
        "topic": "#607D8B",
    }
    default_color = "#9E9E9E"

    # Legend HTML
    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:12px;">'
    sorted_types = sorted(type_colors.items())
    for t, c in sorted_types:
        legend_html += f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:{c};display:inline-block;"></span>{t}</span>'
    legend_html += f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:{default_color};display:inline-block;"></span>other</span>'
    if total_nodes > len(nodes):
        legend_html += f'<span style="color:#666;margin-left:8px;">(showing {len(nodes)} of {total_nodes} nodes)</span>'
    legend_html += '</div>'

    node_items = []
    for n in nodes:
        color = type_colors.get(n["type"].lower(), default_color)
        node_items.append(json.dumps({
            "id": n["id"],
            "label": n["id"],
            "title": f"Type: {n['type']}\nScope: {n['scope']}",
            "color": {"background": color, "border": color, "highlight": {"background": "#FFD700", "border": "#FFD700"}},
            "font": {"size": 12, "color": "#ccc"},
        }))

    edge_items = []
    for e in edges:
        edge_items.append(json.dumps({
            "from": e["source"],
            "to": e["target"],
            "label": e["relation"],
            "arrows": "to",
            "font": {"size": 9, "align": "middle", "color": "#888", "strokeWidth": 0},
            "color": {"color": "#555", "opacity": 0.5},
            "smooth": {"type": "continuous"},
            "width": 1,
        }))

    nodes_json = ",\n    ".join(node_items)
    edges_json = ",\n    ".join(edge_items)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Argos Graph</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#111; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:#ccc; overflow:hidden; }}
  #header {{
    background:#1a1a2e; padding:10px 20px;
    display:flex; justify-content:space-between; align-items:center;
    border-bottom:1px solid #333; flex-wrap:wrap; gap:8px;
    position:relative; z-index:10;
  }}
  #header h1 {{ font-size:15px; font-weight:600; color:#eee; }}
  #stats {{ display:flex; gap:16px; }}
  #stats .stat {{ text-align:center; }}
  #stats .num {{ font-size:16px; font-weight:700; color:#4fc3f7; }}
  #stats .label {{ font-size:9px; color:#666; text-transform:uppercase; letter-spacing:0.5px; }}
  #legend {{ padding:6px 20px; background:#16213e; border-bottom:1px solid #333; position:relative; z-index:10; }}
  #controls {{ padding:6px 20px; background:#16213e; border-bottom:1px solid #333; display:flex; gap:12px; align-items:center; font-size:12px; position:relative; z-index:10; }}
  #controls button, #controls input {{
    background:#2a2a4a; color:#ccc; border:1px solid #444; border-radius:4px;
    padding:4px 10px; font-size:11px; cursor:pointer;
  }}
  #controls button:hover {{ background:#3a3a5a; }}
  #controls input {{ flex:1; max-width:300px; background:#1a1a2e; outline:none; }}
  #controls input:focus {{ border-color:#4fc3f7; }}
  #status {{ color:#888; font-size:11px; }}
  #mynetwork {{ width:100%; height:calc(100vh - 110px); background:#111; }}
</style>
</head>
<body>
<div id="header">
  <h1>Argos Knowledge Graph</h1>
  <div id="stats">
    <div class="stat"><div class="num">{len(nodes)}</div><div class="label">Nodes</div></div>
    <div class="stat"><div class="num">{len(edges)}</div><div class="label">Edges</div></div>
  </div>
</div>
<div id="legend">{legend_html}</div>
<div id="controls">
  <input id="search" type="text" placeholder="Search node..." oninput="filterNodes(this.value)">
  <button onclick="togglePhysics()" id="physBtn">Enable Physics (slow)</button>
  <button onclick="resetView()">Reset View</button>
  <span id="status">Ready</span>
</div>
<div id="mynetwork"></div>

<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<script>
const nodes = new vis.DataSet([
    {nodes_json}
]);
const edges = new vis.DataSet([
    {edges_json}
]);

const container = document.getElementById('mynetwork');
const data = {{ nodes, edges }};
let physicsOn = false;

const options = {{
  physics: {{
    enabled: false,
    solver: 'barnesHut',
    barnesHut: {{ gravitationalConstant:-3000, centralGravity:0.3, springLength:200, springConstant:0.04, damping:0.09 }},
    stabilization: {{ iterations:100 }},
  }},
  layout: {{
    improvedLayout: true,
    randomSeed: 42,
  }},
  interaction: {{
    hover: true, tooltipDelay: 200, navigationButtons: true,
    keyboard: true, zoomView: true, dragView: true,
  }},
  edges: {{
    smooth: {{ type:'continuous' }}, width:1, selectionWidth:2,
  }},
  nodes: {{
    shape:'dot', size:12,
    font:{{ color:'#bbb', size:11, face:'Segoe UI' }},
    borderWidth:1.5,
  }},
}};

const network = new vis.Network(container, data, options);

function togglePhysics() {{
  physicsOn = !physicsOn;
  network.setOptions({{ physics: {{ enabled: physicsOn }} }});
  document.getElementById('physBtn').textContent = physicsOn ? 'Disable Physics' : 'Enable Physics (slow)';
  document.getElementById('status').textContent = physicsOn ? 'Physics running...' : 'Ready';
}}

function resetView() {{
  network.fit({{ animation: true }});
}}

function filterNodes(query) {{
  if (!query) {{
    nodes.forEach(n => nodes.update({{ id: n.id, hidden: false }}));
    edges.forEach(e => edges.update({{ id: e.id, hidden: false }}));
    document.getElementById('status').textContent = 'Showing all';
    return;
  }}
  const q = query.toLowerCase();
  let visible = 0;
  nodes.forEach(n => {{
    const match = n.id.toLowerCase().includes(q);
    nodes.update({{ id: n.id, hidden: !match }});
    if (match) visible++;
  }});
  edges.forEach(e => {{
    const srcMatch = e.from?.toLowerCase().includes(q);
    const tgtMatch = e.to?.toLowerCase().includes(q);
    edges.update({{ id: e.id, hidden: !(srcMatch || tgtMatch) }});
  }});
  document.getElementById('status').textContent = "Showing " + visible + "/" + nodes.length + " nodes";
}}

// Click to focus
network.on('click', function(params) {{
  if (params.nodes.length > 0) {{
    network.focus(params.nodes[0], {{ scale:2, animation:true }});
    document.getElementById('search').value = params.nodes[0];
    filterNodes(params.nodes[0]);
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(Path(output_path).resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the Argos Kuzu graph as an interactive HTML page."
    )
    parser.add_argument("--output", "-o", default="graph_viz.html")
    parser.add_argument("--limit", "-l", type=int, default=200,
                        help="Max nodes (default: 200, use higher for full view)")
    parser.add_argument("--home", type=str, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    home = Path(args.home) if args.home else _get_hermes_home()
    print(f"  HERMES_HOME: {home}")
    print(f"  Loading graph (up to {args.limit} nodes)...")

    nodes, edges, total_nodes, total_edges = load_graph(home, args.limit)
    print(f"  Loaded {len(nodes)} nodes, {len(edges)} edges")
    print(f"  Total in DB: {total_nodes} nodes, {total_edges} edges")

    if len(nodes) == 0:
        print("  Graph is empty.")
        return

    output = generate_html(nodes, edges, total_nodes, total_edges, args.output)
    print(f"  Saved to: {output}")

    full_path = Path(output).resolve()
    if not args.no_browser:
        print("  Opening in browser...")
        webbrowser.open(full_path.as_uri())
    else:
        print(f"  Open: file://{full_path.as_posix()}")


if __name__ == "__main__":
    main()