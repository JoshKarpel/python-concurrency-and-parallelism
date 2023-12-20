from counterweight.cell_paint import CellPaint
from counterweight.styles.styles import CellStyle, Color
from more_itertools import grouper
from structlog import get_logger

logger = get_logger()

black = Color.from_name("black")


def canvas(
    width: int,
    height: int,
    cells: dict[tuple[int, int], Color],
) -> list[CellPaint]:
    c = []
    for y_top, y_bot in grouper(range(height), 2):
        for x in range(width):
            c.append(
                CellPaint(
                    char="▀",
                    style=CellStyle(
                        foreground=cells.get((x, y_top), black),
                        background=cells.get((x, y_bot), black),
                    ),
                )
            )
        c.append(CellPaint(char="\n"))
    return c[:-1]  # strip off last newline


def clamp(min_: int, val: int, max_: int) -> int:
    return max(min_, min(val, max_))
