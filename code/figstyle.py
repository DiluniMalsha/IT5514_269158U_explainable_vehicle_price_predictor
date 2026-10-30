"""Shared chart style, so every figure in the report looks like the same document.

Kept in one place for the same reason `features.py` is: two copies of a palette
drift, and a reader notices when figure 4 and figure 11 disagree about what blue
means.

Conventions these helpers enforce:
  * one series per chart gets one colour; a second colour marks emphasis or a
    genuine second category, never a value ramp over nominal categories
  * gridlines are solid hairlines behind the marks, never dashed
  * no dual-axis charts - two measures means two panels
  * titles are the claim the figure supports, left-aligned to the figure edge

The palette is the validated reference set: categorical slots in fixed order,
diverging blue<->red with a neutral grey midpoint. Checked with a contrast and
colour-vision-deficiency validator rather than by eye.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# categorical slots 1 and 2, plus the diverging poles
BLUE = '#2a78d6'
ORANGE = '#eb6834'
RED = '#e34948'
MID = '#f0efec'

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASE = '#c3c2b7'

DIVERGING = LinearSegmentedColormap.from_list('blue_grey_red', [BLUE, MID, RED])
DIVERGING.set_bad(SURFACE)


def style() -> None:
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'normal',
        'axes.titlecolor': INK2,
        'axes.titlelocation': 'left',
        'axes.labelsize': 9,
        'axes.labelcolor': INK2,
        'axes.edgecolor': BASE,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2,
        'legend.frameon': False,
        'legend.fontsize': 8.5,
        'legend.labelcolor': INK2,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'grid.linestyle': '-',
        'axes.axisbelow': True,
    })


def title(fig, text: str, y: float = 1.02, ax=None) -> None:
    """The figure's claim, left-aligned to the plot area.

    Pass `ax` (the leftmost axes) whenever the chart has long y-tick labels. With
    `bbox_inches='tight'` the saved image grows leftward to fit those labels, so a
    title anchored to figure x=0 ends up floating somewhere in the middle of the
    image while a title on a chart with short labels sits flush left - the same
    code gives two different alignments. Anchoring to the axes keeps every figure
    in the report aligned to the same vertical.

    Call it last, after ticks and any tight_layout, or the position is stale.
    """
    x = ax.get_position().x0 if ax is not None else 0.0
    fig.suptitle(text, x=x, ha='left', fontsize=11, color=INK, y=y)


def multiplier_ticks(ax, axis: str = 'x') -> None:
    """Relabel a log10-contribution axis with price multipliers, adaptively.

    A fixed tick list does not survive terms of different magnitude: `condition`
    spans about +/-0.03 and `age_years` about +/-0.6, so one hard-coded list
    leaves the small panels with a single tick reading x1.00 and nothing else.
    """
    lo, hi = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    span = max(abs(lo), abs(hi))
    for step in (0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.6):
        if span / step <= 4:
            break
    ticks = [t * step for t in range(-4, 5) if abs(t * step) <= span]
    labels = [f'x{10 ** t:.2f}' for t in ticks]
    if axis == 'x':
        ax.set_xticks(ticks, labels, fontsize=8)
    else:
        ax.set_yticks(ticks, labels, fontsize=8)


def vgrid(ax) -> None:
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def hgrid(ax) -> None:
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)


def corner(ax, text: str, loc: str = 'tr') -> None:
    """Plain text in a clear corner. Never a leader line across the data."""
    x, ha = (0.97, 'right') if loc[1] == 'r' else (0.03, 'left')
    y, va = (0.95, 'top') if loc[0] == 't' else (0.05, 'bottom')
    ax.annotate(text, xy=(x, y), xycoords='axes fraction', ha=ha, va=va,
                fontsize=8.5, color=INK2)


def rupees(v, _pos=None) -> str:
    """Axis formatter: rupees in the units the market talks in."""
    if v == 0:
        return '0'
    if abs(v) >= 1e9:
        return f'{v / 1e9:g}B'
    if abs(v) >= 1e6:
        return f'{v / 1e6:g}M'
    return f'{v / 1e3:g}k'


def multiplier_axis(ax, lo: float, hi: float) -> None:
    """Label a log10-contribution axis with price multipliers.

    The model is fitted on log10(price), so a term's contribution *adds* in log
    space and therefore *multiplies* in rupees. Readers think in multipliers, so
    the ticks say x0.5 rather than -0.30. This relabels one axis, it does not add
    a second scale.
    """
    candidates = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 1.0,
                  1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]
    ticks = [np.log10(c) for c in candidates if lo <= np.log10(c) <= hi]
    labels = [f'x{c:g}' for c in candidates if lo <= np.log10(c) <= hi]
    if len(ticks) >= 3:
        ax.set_yticks(ticks, labels)


def save(fig, name: str, figs_dir: Path) -> None:
    figs_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (('png', {'dpi': 200}), ('pdf', {})):
        fig.savefig(figs_dir / f'{name}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)
    print(f'  wrote figures/{name}.png + .pdf')
