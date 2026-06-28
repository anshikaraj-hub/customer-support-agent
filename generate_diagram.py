# Generates the LangGraph workflow diagram for the assignment.
# Run once with: python generate_diagram.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')


def draw_box(ax, x, y, w, h, label, sublabel='', color='#1d4ed8', fontsize=10):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='white',
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center',
                color='white', fontsize=fontsize, fontweight='bold', zorder=4)
        ax.text(x, y - 0.2, sublabel, ha='center', va='center',
                color='#cbd5e1', fontsize=7.5, zorder=4, style='italic')
    else:
        ax.text(x, y, label, ha='center', va='center',
                color='white', fontsize=fontsize, fontweight='bold', zorder=4)


def draw_diamond(ax, x, y, w, h, label, color='#92400e'):
    diamond = plt.Polygon(
        [[x, y + h/2], [x + w/2, y], [x, y - h/2], [x - w/2, y]],
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(diamond)
    ax.text(x, y, label, ha='center', va='center',
            color='white', fontsize=9, fontweight='bold', zorder=4)


def arrow(ax, x1, y1, x2, y2, label='', color='#94a3b8'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=2)
    if label:
        mx, my = (x1 + x2) / 2 + 0.15, (y1 + y2) / 2
        ax.text(mx, my, label, color='#f59e0b', fontsize=8, fontweight='bold', zorder=5)


# ── Title ──
ax.text(7, 19.4, 'ABC Technologies', ha='center',
        color='#93c5fd', fontsize=15, fontweight='bold')
ax.text(7, 18.95, 'AI Customer Support Automation System - LangGraph Workflow',
        ha='center', color='#e2e8f0', fontsize=10)
ax.axhline(18.65, color='#334155', linewidth=1)

# ── Customer Query ──
draw_box(ax, 7, 18.2, 5, 0.55, 'Customer Query', color='#1e3a5f')
arrow(ax, 7, 17.93, 7, 17.53)

# ── Load Memory ──
draw_box(ax, 7, 17.2, 5.5, 0.6, 'Task 7: Load Memory',
         'SQLite - Retrieve conversation history', color='#0f766e')
arrow(ax, 7, 16.9, 7, 16.5)

# ── Intent Classifier ──
draw_box(ax, 7, 16.2, 5.5, 0.6, 'Task 3: Intent Classifier',
         'Llama3.2 - Sales / Technical / Billing / Account / Memory', color='#1d4ed8')
arrow(ax, 7, 15.9, 7, 15.5)

# ── RAG Retrieval ──
draw_box(ax, 7, 15.2, 5.5, 0.6, 'Task 6: RAG Retrieval',
         'TF-IDF Knowledge Base Search', color='#7c3aed')
arrow(ax, 7, 14.9, 7, 14.5)

# ── Conditional Router ──
draw_diamond(ax, 7, 14.1, 5, 0.75, 'Task 4: Conditional Router', color='#92400e')
arrow(ax, 7, 13.73, 7, 13.33)

# ── 5 Agents ──
agents = [
    (1.2,  'Sales\nAgent',     '#15803d'),
    (3.5,  'Technical\nAgent', '#1d4ed8'),
    (7.0,  'Billing\nAgent',   '#c2410c'),
    (10.5, 'Account\nAgent',   '#0f766e'),
    (12.8, 'Memory\nAgent',    '#374151'),
]
for ax_x, lbl, col in agents:
    draw_box(ax, ax_x, 12.9, 2.2, 0.75, lbl, color=col, fontsize=9)
    arrow(ax, 7, 13.73, ax_x, 13.28, color='#f59e0b')

# ── High Risk Check ──
draw_diamond(ax, 7, 11.8, 5, 0.75, 'Task 8: High-Risk Check?', color='#9f1239')
for ax_x, _, _ in agents:
    arrow(ax, ax_x, 12.53, 7, 12.18, color='#94a3b8')

# ── Human Approval ──
draw_box(ax, 3.5, 10.5, 4, 0.7, 'Task 8: Human Approval',
         'Supervisor reviews and decides', color='#b91c1c')
ax.annotate('', xy=(5.5, 10.5), xytext=(5.5, 11.8),
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.8), zorder=2)
ax.text(5.65, 11.15, 'Yes', color='#ef4444', fontsize=8, fontweight='bold')

# ── Skip arrow ──
ax.annotate('', xy=(7, 10.1), xytext=(9.5, 11.8),
            arrowprops=dict(arrowstyle='->', color='#22c55e', lw=1.8), zorder=2)
ax.text(9.0, 11.1, 'No', color='#22c55e', fontsize=8, fontweight='bold')

# ── Supervisor Agent ──
draw_box(ax, 7, 9.7, 5.5, 0.6, 'Task 9: Supervisor Agent',
         'Quality validation and response polish', color='#5b21b6')
arrow(ax, 3.5, 10.15, 7, 10.0, color='#94a3b8')
arrow(ax, 7, 10.4, 7, 10.0)

# ── Save Memory ──
draw_box(ax, 7, 8.7, 5.5, 0.6, 'Task 7: Save Memory',
         'Persist conversation to SQLite', color='#0f766e')
arrow(ax, 7, 9.4, 7, 9.0)

# ── Final Response ──
draw_box(ax, 7, 7.7, 5, 0.55, 'Final Response to Customer', color='#14532d')
arrow(ax, 7, 8.4, 7, 7.98)

# ── Knowledge Base sidebar ──
draw_box(ax, 12.2, 15.5, 2.8, 0.55, 'Knowledge Base', color='#312e81', fontsize=9)
for i, doc in enumerate(['Company Policy', 'Pricing Guide', 'Technical Manual', 'FAQ']):
    draw_box(ax, 12.2, 14.85 - i * 0.6, 2.8, 0.45, doc, color='#1e1b4b', fontsize=8)
ax.annotate('', xy=(9.75, 15.2), xytext=(10.8, 15.2),
            arrowprops=dict(arrowstyle='<-', color='#7c3aed', lw=1.5,
                            linestyle='dashed'), zorder=2)
ax.text(10.1, 15.35, 'RAG', color='#a78bfa', fontsize=8)

# ── SQLite sidebar ──
draw_box(ax, 1.5, 17.2, 2.5, 0.55, 'SQLite DB', 'memory.db', color='#134e4a', fontsize=9)
ax.annotate('', xy=(4.25, 17.2), xytext=(2.75, 17.2),
            arrowprops=dict(arrowstyle='<->', color='#2dd4bf', lw=1.5,
                            linestyle='dashed'), zorder=2)
ax.text(3.1, 17.38, 'R/W', color='#5eead4', fontsize=8)

# ── Legend ──
ax.axhline(6.8, color='#334155', linewidth=0.8)
ax.text(1.5, 6.5, 'Legend:', color='#e2e8f0', fontsize=9, fontweight='bold')
legend_items = [
    ('#1d4ed8', 'LLM Node'),
    ('#15803d', 'Support Agent'),
    ('#7c3aed', 'RAG Retrieval'),
    ('#0f766e', 'Memory (SQLite)'),
    ('#b91c1c', 'Human-in-the-Loop'),
    ('#92400e', 'Conditional Router'),
]
for i, (col, lbl) in enumerate(legend_items):
    x = 1.5 + (i % 3) * 4.2
    y = 6.0 - (i // 3) * 0.7
    box = FancyBboxPatch((x, y - 0.18), 0.4, 0.35,
                         boxstyle="round,pad=0.05", facecolor=col,
                         edgecolor='white', linewidth=1, zorder=3)
    ax.add_patch(box)
    ax.text(x + 0.55, y, lbl, color='#e2e8f0', fontsize=8.5, va='center')

ax.text(7, 4.5,
        'Tasks: 1-Workflow  2-State  3-Intent  4-Routing  5-Agents  '
        '6-RAG  7-Memory  8-Human-in-Loop  9-Supervisor  10-Demo',
        ha='center', color='#64748b', fontsize=7.5)

plt.tight_layout()
plt.savefig('diagrams/workflow_diagram.png', dpi=150,
            bbox_inches='tight', facecolor='#0f172a')
print("Diagram saved to diagrams/workflow_diagram.png")