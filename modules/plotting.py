

import matplotlib
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import modules.coordinates as crd

class HandlerBlackWhiteDashed(HandlerBase):
    """Draw a white dashed line over a thicker black line in the legend."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        y = height / 2 - ydescent
        x = [-xdescent, width - xdescent]

        black = Line2D(
            x, [y, y], color="black", linewidth=3, solid_capstyle="butt", transform=trans
        )
        white = Line2D(
            x, [y, y], color="white", linestyle="--", linewidth=1.5,
            dash_capstyle="butt", transform=trans
        )
        return [black, white]


def add_scalebar_lonlat(ax, length_km=100, fontsize=10, bold=False, location=(0.7, 0.05), label_y_offset=0.001):
    """
    Add a scale bar to a lon/lat map.
    
    ax : matplotlib axis
    length_km : length of scale bar in km
    location : (x_frac, y_frac) in axis coordinates
    """

    # Get map extent
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Center latitude
    lat_c = (ylim[0] + ylim[1]) / 2.0

    # Convert km to degrees longitude at this latitude
    length_deg = crd.km_to_lon(length_km, lat_c)

    # Position in data coords
    x_start = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    y_start = ylim[0] + (ylim[1] - ylim[0]) * location[1]

    # Draw scale bar — clip_on=False ensures it's visible outside the axis
    line, = ax.plot([x_start, x_start + length_deg], [y_start, y_start],
                    color='black', linewidth=3, clip_on=False)

    # Add label — clip_on=False ensures it's visible outside the axis
    text = ax.text(x_start + length_deg / 2, y_start + label_y_offset, f'{length_km} km',
                   ha='center', va='bottom', fontsize=fontsize, fontweight='bold' if bold else 'normal', clip_on=False)
    

def add_north_arrow(target, location=(0.9, 0.9), size=0.08):
    """
    Add a north arrow to a matplotlib axis or figure.

    target   : matplotlib Axes or Figure
    location : (x, y) in axes or figure coordinates (0-1)
    size     : length of the arrow in coordinate units
    """
    x, y = location

    if isinstance(target, matplotlib.figure.Figure):
        trans       = target.transFigure
        add_artist  = target.add_artist
        add_text    = target.text
    elif isinstance(target, matplotlib.axes.Axes):
        trans       = target.transAxes
        add_artist  = target.add_patch
        add_text    = target.text
    else:
        raise TypeError(f"target must be a matplotlib Axes or Figure, got {type(target)}")

    arrow = FancyArrowPatch((x, y - size), (x, y),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='black',
                            transform=trans, clip_on=False)
    add_artist(arrow)

    add_text(x, y - 0.005, 'N', transform=trans,
             ha='center', va='bottom', fontsize=12, fontweight='bold', clip_on=False)