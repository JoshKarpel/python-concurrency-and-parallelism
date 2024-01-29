from counterweight.elements import Chunk
from counterweight.styles.styles import CellStyle, Color
from more_itertools import grouper
from structlog import get_logger

logger = get_logger()

BLACK = Color.from_name("black")


def canvas(
    width: int,
    height: int,
    cells: dict[tuple[int, int], Color],
) -> list[Chunk]:
    c: list[Chunk] = []
    for y_top, y_bot in grouper(range(height), 2):
        c.extend(
            Chunk(
                content="▀",
                style=CellStyle(
                    foreground=cells.get((x, y_top), BLACK),
                    background=cells.get((x, y_bot), BLACK),
                ),
            )
            for x in range(width)
        )
        c.append(Chunk.newline())
    return c[:-1]  # strip off last newline


def clamp(min_: int, val: int, max_: int) -> int:
    return max(min_, min(val, max_))
