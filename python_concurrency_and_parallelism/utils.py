import inspect
import random
from asyncio import sleep
from functools import lru_cache
from itertools import product
from typing import Callable

from counterweight.components import component
from counterweight.elements import Chunk, Div, Text
from counterweight.hooks import use_effect, use_state
from counterweight.styles.styles import COLORS_BY_NAME
from counterweight.styles.utilities import *
from more_itertools import grouper
from pygments.lexers import get_lexer_by_name
from pygments.styles import get_style_by_name
from structlog import get_logger

logger = get_logger()

full_block = "█"

BLACK = Color.from_name("black")

python_blue = Color.from_hex("#4B8BBE")
python_chunk = Chunk(content="Python", style=CellStyle(foreground=python_blue))

palette = [Color.from_hex(c) for c in ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d")]


def clamp(min_: int, val: int, max_: int) -> int:
    return max(min_, min(val, max_))


def drop_shadow(*c: Text | Div) -> Div:
    return Div(
        style=border_lightshade | border_bottom_right | border_contract_1 | border_gray_400 | weight_none,
        children=list(c),
    )


def colored_bar(*blocks: tuple[int, Color]) -> list[Chunk]:
    return [Chunk(content=full_block * n, style=CellStyle(foreground=c)) for n, c in blocks]


def time_arrow(length: int) -> Text:
    return Text(
        content="Time " + ("─" * length) + "→",
        style=weight_none,
    )


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
    c.pop()  # strip off last newline
    return c


moves = [(x, y) for x, y in product((-1, 0, 1), repeat=2) if (x, y) != (0, 0)]


@component
def random_walkers(width: int, height: int, threads: int, first: bool) -> Text:
    colors, set_colors = use_state(random.sample(list(COLORS_BY_NAME.values()), k=threads))
    walkers, set_walkers = use_state([(random.randrange(width), random.randrange(height)) for _ in range(len(colors))])

    def update_movers(m: list[tuple[int, int]]) -> list[tuple[int, int]]:
        new = []
        for x, y in m:
            dx, dy = random.choice(moves)
            new.append((clamp(0, x + dx, width - 1), clamp(0, y + dy, height - 1)))
        return new

    async def tick() -> None:
        while True:
            await sleep(0.5)
            set_walkers(update_movers)

    use_effect(tick, deps=())

    return Text(
        content=(
            canvas(
                width=width,
                height=height,
                cells=dict(zip(walkers, colors)),
            )
        ),
        style=border_heavy | border_slate_400 | (border_top_bottom_right if not first else None) | weight_none,
    )


@lru_cache(maxsize=2**12)
def style_for_token(style: str, token: object) -> CellStyle:
    s = get_style_by_name(style).style_for_token(token)

    return CellStyle(
        foreground=Color.from_hex(s.get("color") or "000000"),
        background=Color.from_hex(s.get("bgcolor") or "000000"),
        bold=s["bold"],
        italic=s["italic"],
        underline=s["underline"],
    )


def make_code_example(*fns: Callable[[...], object] | str) -> Div:
    lexer = get_lexer_by_name("python")

    chunks = []
    for fn in fns:
        for token, text in lexer.get_tokens(inspect.getsource(fn) if callable(fn) else fn):
            s = style_for_token("github-dark", token)
            chunks.append(
                Chunk(
                    content=text,
                    style=s,
                )
            )
        chunks.append(Chunk.newline())

    # pygments seems to add a trailing newline even if removed from the source,
    # probably good for files but not for slides, so pop it off
    chunks.pop()
    chunks.pop()

    return Div(
        style=col | align_self_stretch | align_children_center | justify_children_center,
        children=[
            Text(
                style=weight_none
                | pad_x_1
                | border_heavy
                | Style(border=Border(style=CellStyle(foreground=python_blue))),
                content=chunks,
            ),
        ],
    )


def make_activity_bars(
    buckets: int,
    offset: int,
    thread_results: list[list[int]],
) -> list[Text]:
    biggest_count = max(max(tracker[offset : offset + buckets]) for tracker in thread_results) or 1
    return [
        Text(
            content=[
                Chunk(content=f"   {n} "),
                *colored_bar(
                    *((1, BLACK.blend(palette[n], t / biggest_count)) for t in tracker[offset : offset + buckets])
                ),
                Chunk.space(),
                Chunk(content=f"{sum(tracker[offset:offset + buckets]):>6}"),
            ],
            style=weight_none,
        )
        for n, tracker in enumerate(thread_results)
    ] + [time_arrow(10)]
