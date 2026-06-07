"""
Generate docs/multi_agent_flow.png — LangGraph multi-agent architecture diagram.
Run: python docs/generate_flow.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Canvas ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Colour palette ────────────────────────────────────────────
CLR = {
    "user":       "#1f6feb",
    "orch":       "#8b5cf6",
    "graph":      "#1a2332",
    "graph_border":"#30363d",
    "classify":   "#e879f9",
    "rag":        "#22c55e",
    "billing":    "#f59e0b",
    "hr":         "#06b6d4",
    "general":    "#94a3b8",
    "service":    "#16213e",
    "service_border": "#1f6feb",
    "arrow":      "#6e7681",
    "arrow_hi":   "#58a6ff",
    "text_main":  "#e6edf3",
    "text_dim":   "#8b949e",
    "text_code":  "#79c0ff",
    "end":        "#374151",
}


def box(ax, x, y, w, h, color, border, radius=0.35, alpha=1.0, lw=1.5):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=border,
                       facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(p)


def label(ax, x, y, text, size=10, color="#e6edf3", weight="normal",
          ha="center", va="center", family="monospace"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, fontfamily=family, zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#6e7681", lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)


def arrow_curve(ax, x1, y1, x2, y2, color, rad=0.2, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=2)


# ══════════════════════════════════════════════════════════════
# 1.  USER
# ══════════════════════════════════════════════════════════════
box(ax, 9, 12, 4.4, 0.9, CLR["user"], "#58a6ff", lw=2)
label(ax, 9, 12.0, "User  /  Client", size=12, weight="bold", color="#ffffff")
label(ax, 9, 11.62, 'POST /agent   {"message": "..."}',
      size=8.5, color=CLR["text_code"])

# ══════════════════════════════════════════════════════════════
# 2.  FastAPI server
# ══════════════════════════════════════════════════════════════
box(ax, 9, 10.8, 4.4, 0.9, "#161b22", "#30363d", lw=1.5)
label(ax, 9, 11.02, "api/server.py  ·  FastAPI :8002", size=10,
      weight="bold", color=CLR["text_main"])
label(ax, 9, 10.62, "enriches message with tenant_id / user_id",
      size=8, color=CLR["text_dim"])
arrow(ax, 9, 11.55, 9, 11.25, color=CLR["arrow_hi"], lw=2)

# ══════════════════════════════════════════════════════════════
# 3.  LangGraph boundary
# ══════════════════════════════════════════════════════════════
lg_box = FancyBboxPatch((0.6, 6.2), 16.8, 3.9,
                        boxstyle="round,pad=0,rounding_size=0.5",
                        linewidth=1.5, linestyle="--",
                        edgecolor=CLR["orch"], facecolor=CLR["graph"],
                        alpha=0.55, zorder=1)
ax.add_patch(lg_box)
label(ax, 1.8, 9.88, "LangGraph  StateGraph", size=9,
      color=CLR["orch"], weight="bold")
label(ax, 3.35, 9.55, "orchestrator/graph.py", size=7.5,
      color=CLR["text_dim"])

arrow(ax, 9, 10.35, 9, 9.95, color=CLR["arrow_hi"], lw=2)

# ══════════════════════════════════════════════════════════════
# 4.  classify_node
# ══════════════════════════════════════════════════════════════
box(ax, 9, 9.45, 5.2, 0.85, "#2d1557", CLR["classify"], lw=2, radius=0.3)
label(ax, 9, 9.65, "classify_node", size=11, weight="bold", color="#f0abfc")
label(ax, 9, 9.28, "GPT-4o  ·  tool_choice='required'  ·  picks specialist",
      size=8, color=CLR["text_dim"])

# ══════════════════════════════════════════════════════════════
# 5.  conditional fan-out
# ══════════════════════════════════════════════════════════════
SPECIALISTS = [
    (2.8,  7.65, "rag_memory_node",  "RagMemoryAgent",    CLR["rag"],     "#16a34a", "rag-memory"),
    (6.6,  7.65, "billing_node",     "BillingAgent",      CLR["billing"], "#d97706", "billing"),
    (10.4, 7.65, "hr_node",          "HRAgent",           CLR["hr"],      "#0891b2", "hr"),
    (14.2, 7.65, "general_node",     "GeneralAgent",      CLR["general"], "#475569", "general"),
]

for sx, sy, node, agent, fill, border, _ in SPECIALISTS:
    arrow_curve(ax, 9, 9.03, sx, sy + 0.43, color=CLR["classify"], rad=0.0, lw=1.8)

for sx, sy, node, agent, fill, border, _ in SPECIALISTS:
    box(ax, sx, sy, 3.6, 0.85, fill + "33", border, lw=2, radius=0.3)
    label(ax, sx, sy + 0.18, node, size=9, weight="bold", color=border)
    label(ax, sx, sy - 0.18, agent, size=8, color=CLR["text_dim"])

# ══════════════════════════════════════════════════════════════
# 6.  END node
# ══════════════════════════════════════════════════════════════
box(ax, 9, 6.55, 2.4, 0.55, CLR["end"], "#6e7681", radius=0.28)
label(ax, 9, 6.55, "END", size=10, weight="bold", color="#9ca3af")

for sx, sy, *_ in SPECIALISTS:
    arrow(ax, sx, sy - 0.43, 9, 6.83, color=CLR["arrow"], lw=1.2)

# ══════════════════════════════════════════════════════════════
# 7.  executor/runner.py
# ══════════════════════════════════════════════════════════════
box(ax, 9, 5.6, 4.8, 0.82, "#161b22", "#30363d", lw=1.5)
label(ax, 9, 5.82, "executor/runner.py", size=10, weight="bold",
      color=CLR["text_main"])
label(ax, 9, 5.44, "httpx  ·  GET / POST / DELETE / PUT  ·  handles direct_answer",
      size=7.8, color=CLR["text_dim"])
arrow(ax, 9, 6.28, 9, 6.0, color=CLR["arrow_hi"], lw=2)

# ══════════════════════════════════════════════════════════════
# 8.  Downstream services
# ══════════════════════════════════════════════════════════════
SERVICES = [
    (4.5, 4.1, ":8001  RAG Memory System",
     "POST /api/rag/query\nPOST /api/rag/ingest/text\nDELETE /api/memory/erase",
     CLR["rag"], "#16a34a"),
    (13.5, 4.1, "Direct Answer",
     "billing / hr / general\nno HTTP call needed\ndirect_answer field",
     "#f59e0b", "#d97706"),
]

for dx, dy, title, sub, fill, border in SERVICES:
    box(ax, dx, dy, 6.6, 1.6, fill + "15", border, lw=2, radius=0.4)
    label(ax, dx, dy + 0.52, title, size=10, weight="bold", color=border)
    for i, line in enumerate(sub.split("\n")):
        label(ax, dx, dy + 0.08 - i * 0.38, line, size=8, color=CLR["text_dim"])

arrow_curve(ax, 7.2, 5.2, 4.5, 4.9, color=CLR["rag"], rad=-0.2, lw=1.8)
arrow_curve(ax, 10.8, 5.2, 13.5, 4.9, color=CLR["billing"], rad=0.2, lw=1.8)

# ══════════════════════════════════════════════════════════════
# 9.  Response path (dashed)
# ══════════════════════════════════════════════════════════════
ax.annotate("", xy=(9, 3.4), xytext=(9, 5.19),
            arrowprops=dict(arrowstyle="<-", color="#58a6ff",
                            lw=1.5, linestyle="dashed"), zorder=2)
label(ax, 10.6, 4.25, "AgentReply JSON", size=8,
      color="#58a6ff", weight="bold")

# ══════════════════════════════════════════════════════════════
# 10.  Legend
# ══════════════════════════════════════════════════════════════
legend_items = [
    (CLR["classify"],  CLR["orch"],    "LangGraph classify node (GPT-4o)"),
    (CLR["rag"]+"33",  CLR["rag"],     "rag-memory specialist"),
    (CLR["billing"]+"33", CLR["billing"], "billing specialist"),
    (CLR["hr"]+"33",   CLR["hr"],      "hr specialist"),
    (CLR["general"]+"33", CLR["general"], "general specialist"),
]
lx, ly = 0.9, 2.9
for fill, border, txt in legend_items:
    p = FancyBboxPatch((lx, ly - 0.12), 0.38, 0.28,
                       boxstyle="round,pad=0,rounding_size=0.08",
                       linewidth=1.2, edgecolor=border, facecolor=fill, zorder=5)
    ax.add_patch(p)
    ax.text(lx + 0.52, ly + 0.02, txt, fontsize=8,
            color=CLR["text_dim"], va="center", zorder=5)
    ly -= 0.5

# Title
label(ax, 9, 0.65,
      "Multi-Agent System  ·  LangGraph StateGraph Orchestrator  ·  4 Specialists",
      size=12, weight="bold", color=CLR["text_main"])
label(ax, 9, 0.28,
      "classify_node routes via GPT-4o function calling  ·  rag-memory calls :8001  ·  billing/hr/general answer directly",
      size=8, color=CLR["text_dim"])

# ══════════════════════════════════════════════════════════════
out = os.path.join(os.path.dirname(__file__), "multi_agent_flow.png")
plt.tight_layout(pad=0)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
