"""Render the repository overview diagram used in the README.

The figure has two panels: the ellipsoidal viewing manifold that parameterises a scan,
and the order in which the acquisition, reconstruction, registration and evaluation
stages run. Both panels are drawn from the trajectory equations and the stage graph,
so they are schematics of the software rather than plots of measured data.

Two colour variants are written so that GitHub can pick one per colour scheme:

    python tools/render_diagrams.py --output assets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parents[1]

# One typeface and one type size across both panels.
STYLE = {
    "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "font.family": "serif", "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

THEMES = {
    "light": {"fg": "#1f2328", "muted": "#57606a", "wire": "#8c959f", "accent": "#1f77b4"},
    "dark": {"fg": "#e6edf3", "muted": "#9198a1", "wire": "#6e7681", "accent": "#58a6ff"},
}

STAGES = [
    "MoveIt 2 / Isaac Sim\nCollision checks and RGB acquisition",
    "View assembly / VGGT\nSparse views and optional global context",
    "OBB (optional) / BUFFER-X x2\nReference-assisted registration",
    "F1 / C2M / HD95\nQuality and acquisition-path length",
]


def draw_pipeline(fig, rect: list[float], theme: dict) -> None:
    """Draw the four processing stages and the two reference-geometry branches."""
    fg, muted = theme["fg"], theme["muted"]
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4.30); ax.axis("off")
    levels = [3.22, 2.24, 1.26, .28]
    for y, label in zip(levels, STAGES):
        ax.add_patch(FancyBboxPatch((.08, y), 3.18, .60, boxstyle="round,pad=.025,rounding_size=.06",
                                    fill=False, linewidth=.9, edgecolor=fg))
        ax.text(1.67, y + .30, label, ha="center", va="center", linespacing=1.45, color=fg)
    for upper, lower in zip(levels[:-1], levels[1:]):
        ax.add_patch(FancyArrowPatch((1.67, upper - .02), (1.67, lower + .64), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=.9, color=fg))
    ax.text(1.55, 3.03, "completion above 75%", ha="right", va="center", color=muted)
    ax.text(.03, 4.24, "Six-parameter trajectory grid", va="top", color=fg)
    ax.plot([3.28, 3.60, 3.60], [3.52, 3.52, .58], linewidth=.9, linestyle="--", color=muted)
    for y in (1.56, .58):
        ax.add_patch(FancyArrowPatch((3.60, y), (3.28, y), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=.9, color=muted))
    ax.text(3.83, 2.16, "Known reference geometry", rotation=90, ha="center", va="center", color=muted)


def draw_manifold(fig, rect: list[float], theme: dict) -> None:
    """Draw the ellipsoidal viewing manifold with three illustrative scan arcs."""
    fg, muted, wire, accent = theme["fg"], theme["muted"], theme["wire"], theme["accent"]
    r, z = .22, .50
    tt, pp = np.meshgrid(np.linspace(0, np.pi/2, 19), np.linspace(-np.pi, np.pi, 37))
    ax = fig.add_axes(rect, projection="3d")
    ax.plot_wireframe(r*np.sin(tt)*np.cos(pp), r*np.sin(tt)*np.sin(pp), z*np.cos(tt),
                      rstride=3, cstride=3, linewidth=.45, alpha=.45, color=wire)

    path = []
    angles = np.deg2rad([-140, -10, 120])
    for index, phi in enumerate(angles):
        theta = np.deg2rad(np.linspace(15, 70, 6))
        if index % 2:
            theta = theta[::-1]
        path.extend(np.column_stack([r*np.sin(theta)*np.cos(phi),
                                     r*np.sin(theta)*np.sin(phi), z*np.cos(theta)]))
    path = np.array(path)
    ax.plot(*path.T, "-o", linewidth=1.8, markersize=3.2, color=accent)
    for i in (1, 7, 13):
        step = path[i+1] - path[i]
        ax.quiver(*(path[i] + .25*step), *(step*.45), arrow_length_ratio=.5,
                  linewidth=1.2, color=accent)
    for i in (2, 8, 15):
        ax.plot([path[i, 0], 0], [path[i, 1], 0], [path[i, 2], 0],
                linestyle="--", linewidth=.7, alpha=.85, color=muted)
        ax.quiver(*path[i], *(-path[i]*.17), arrow_length_ratio=.45, linewidth=1, color=muted)

    ax.scatter([0], [0], [0], s=24, marker="x", color=fg)
    ax.text(.012, -.018, -.050, r"$\mathbf{c}$", color=fg)
    ax.quiver(0, 0, 0, 0, 0, z, arrow_length_ratio=.08, linewidth=1, color=fg)
    ax.quiver(0, 0, 0, .22, 0, 0, arrow_length_ratio=.14, linewidth=1, color=fg)
    ax.text(.135, -.018, -.038, r"$r_{xy}$", color=fg)
    arc = np.linspace(angles[0], angles[-1], 100)
    ax.plot(.252*np.cos(arc), .252*np.sin(arc), np.zeros_like(arc),
            linestyle="--", linewidth=.9, color=muted)
    ax.text(.285, .075, -.012, r"$\Delta\phi$", color=fg)

    # Schematic target exclusion volume; its dimensions are illustrative only.
    a, lo, hi = .046, -.04, .035
    for x in (-a, a):
        for y in (-a, a):
            ax.plot([x, x], [y, y], [lo, hi], linewidth=.7, alpha=.85, color=muted)
    for h in (lo, hi):
        ax.plot([-a, a, a, -a, -a], [-a, -a, a, a, -a], [h]*5,
                linewidth=.7, alpha=.85, color=muted)

    ax.set_box_aspect((.70, .70, .68)); ax.view_init(elev=19, azim=-54)
    ax.set_xlim(-.255, .255); ax.set_ylim(-.255, .255); ax.set_zlim(-.055, .505)
    ax.set_axis_off()
    ax.text2D(.02, .99, "Ellipsoidal viewing manifold", transform=ax.transAxes, va="top", color=fg)
    callouts = [("$r_z$", [0, 0, .31], (.815, .695)),
                (r"$\theta_{\min}$", path[0], (.245, .845)),
                (r"$\theta_{\max}$", path[5], (.055, .375)),
                (r"$\phi_{\min}$", path[3], (.045, .615)),
                (r"$\phi_{\max}$", path[-3], (.855, .475))]
    for label, position, location in callouts:
        x2, y2, _ = proj3d.proj_transform(*position, ax.get_proj())
        ax.annotate(label, (x2, y2), xytext=location, textcoords="axes fraction",
                    ha="center", va="center", color=fg,
                    arrowprops={"arrowstyle": "-", "lw": .7, "color": fg})


def render(out: Path, variant: str) -> Path:
    theme = THEMES[variant]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 3.15))
        draw_manifold(fig, [.005, .085, .475, .90], theme)
        draw_pipeline(fig, [.515, .085, .475, .90], theme)
        fig.text(.2425, .015, "(a) Camera-viewing geometry", ha="center", color=theme["fg"])
        fig.text(.7525, .015, "(b) Acquisition and evaluation pipeline", ha="center", color=theme["fg"])
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"overview-{variant}.png"
        fig.savefig(target, dpi=220, transparent=True)
        plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    for variant in THEMES:
        print("wrote", render(args.output, variant))


if __name__ == "__main__":
    main()
