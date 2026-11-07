

from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

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