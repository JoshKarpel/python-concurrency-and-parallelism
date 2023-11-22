import asyncio
import random
from asyncio import sleep
from datetime import datetime
from itertools import product

from reprisal.app import app
from reprisal.components import Chunk, Div, Text, component
from reprisal.events import KeyPressed
from reprisal.hooks import use_effect, use_state
from reprisal.keys import Key
from reprisal.styles.styles import COLORS_BY_NAME
from reprisal.styles.utilities import *
from structlog import get_logger

from python_concurrency_and_parallelism.utils import canvas, clamp

logger = get_logger()

full_block = "█"

python_blue = Color.from_hex("#4B8BBE")
python_chunk = Chunk(content="Python", style=CellStyle(foreground=python_blue))


@component
def root() -> Div:
    slides = [
        title,
        you_may_have_heard,
        definitions,
        processes_and_threads_in_memory,
    ]

    current_slide, set_current_slide = use_state(0)

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Right:
            set_current_slide(lambda n: clamp(0, n + 1, len(slides) - 1))
        elif event.key == Key.Left:
            set_current_slide(lambda n: clamp(0, n - 1, len(slides) - 1))

    return Div(
        style=col,
        children=[
            Div(
                style=col | align_self_stretch,
                children=[slides[current_slide]()],
            ),
            footer(current_slide=current_slide + 1, total_slides=len(slides)),
        ],
        on_key=on_key,
    )


@component
def footer(current_slide: int, total_slides: int) -> Div:
    current_time, set_current_time = use_state(datetime.now())

    async def tick() -> None:
        while True:
            await sleep(1 / 60)
            set_current_time(datetime.now())

    use_effect(tick, deps=())

    return Div(
        style=row
        | justify_children_space_between
        | weight_none
        | align_self_stretch
        | border_slate_400
        | border_light
        | border_top,
        children=[
            Text(
                content=[
                    Chunk(
                        content="Concurrency & Parallelism in Python",
                    ),
                ],
                style=text_slate_200,
            ),
            Text(content=f"{current_time:%Y-%m-%d %I:%M %p}", style=text_slate_200),
            Text(
                content=[
                    Chunk(
                        content=f"{current_slide}",
                    ),
                    Chunk(
                        content=" / ",
                    ),
                    Chunk(
                        content=f"{total_slides}",
                    ),
                ],
                style=text_slate_200,
            ),
        ],
    )


@component
def title() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            Text(content="Concurrency & Parallelism in Python", style=weight_none),
            Text(content="Josh Karpel", style=weight_none),
            Text(content="MadPy, May 2024", style=weight_none | text_gray_400),
        ],
    )


palette = [Color.from_hex(c) for c in ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d")]


def colored_bar(*blocks: tuple[int, int]) -> list[Chunk]:
    return [Chunk(content=full_block * n, style=CellStyle(foreground=palette[c])) for n, c in blocks]


def time_arrow(length: int) -> Text:
    return Text(
        content="Time " + ("─" * length) + "→",
        style=weight_none,
    )


@component
def definitions() -> Div:
    arrow_shift, set_arrow_shift = use_state(0)

    arrow = Chunk(content=" " * arrow_shift + "↑", style=CellStyle(foreground=python_blue))

    async def tick() -> None:
        while True:
            await sleep(0.25)
            set_arrow_shift(lambda n: (n + 1) % 20)

    use_effect(tick, deps=())

    half_and_half_div_style = col | align_children_center
    color_bar_div_style = col | border_heavy | border_gray_400 | pad_1

    concurrency = Div(
        # TODO: gap_children_N is broken here?
        style=half_and_half_div_style,
        children=[
            Text(
                content=[Chunk(content="Concurrency", style=CellStyle(underline=True))],
                style=weight_none | pad_bottom_1,
            ),
            Text(
                content=[
                    Chunk(content="Tasks are"),
                    Chunk.space(),
                    Chunk(content="interleaved", style=CellStyle(foreground=green_600)),
                ],
                style=weight_none | pad_bottom_1,
            ),
            Div(
                style=color_bar_div_style,
                children=[
                    Text(
                        content=colored_bar((3, 0), (2, 1), (1, 2), (5, 0), (2, 2), (3, 3), (1, 1), (3, 0)),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    time_arrow(14),
                ],
            ),
        ],
    )
    parallelism = Div(
        style=half_and_half_div_style,
        children=[
            Text(
                content=[Chunk(content="Parallelism", style=CellStyle(underline=True))],
                style=weight_none | pad_bottom_1,
            ),
            Text(
                content=[
                    Chunk(content="Tasks are"),
                    Chunk.space(),
                    Chunk(content="simultaneous", style=CellStyle(foreground=green_600)),
                ],
                style=weight_none | pad_bottom_1,
            ),
            Div(
                style=color_bar_div_style,
                children=[
                    Text(
                        content=colored_bar((20, 0)),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar((11, 1), (9, 5)),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar((14, 2), (6, 6)),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar((20, 3)),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    time_arrow(14),
                ],
            ),
        ],
    )

    return Div(
        style=row | align_self_stretch | align_children_center | justify_children_space_around,
        children=[concurrency, parallelism],
    )


@component
def you_may_have_heard() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            Text(content="Some things you may have heard...", style=weight_none | pad_bottom_2),
            Text(
                content=[
                    Chunk(content="The"),
                    Chunk.space(),
                    Chunk(content="Global Interpreter Lock (GIL)", style=CellStyle(foreground=red_500)),
                    Chunk.space(),
                    Chunk(content="makes"),
                    Chunk.space(),
                    python_chunk,
                    Chunk.space(),
                    Chunk(content="slow", style=CellStyle(foreground=red_500, underline=True)),
                ],
                style=weight_none,
            ),
            Text(
                content=[
                    Chunk(content="The"),
                    Chunk.space(),
                    Chunk(content="GIL", style=CellStyle(foreground=red_500)),
                    Chunk.space(),
                    Chunk(content="makes"),
                    Chunk.space(),
                    Chunk(content="mulithreading", style=CellStyle(foreground=green_600)),
                    Chunk.space(),
                    Chunk(content="useless", style=CellStyle(foreground=red_500, underline=True)),
                    Chunk.space(),
                    Chunk(content="in"),
                    Chunk.space(),
                    python_chunk,
                ],
                style=weight_none,
            ),
            Text(
                content=[
                    Chunk(content="(but that"),
                    Chunk.space(),
                    Chunk(content="doesn't matter", style=CellStyle(underline=True)),
                    Chunk(content=", just use"),
                    Chunk.space(),
                    Chunk(content="multiprocessing", style=CellStyle(foreground=green_600)),
                    Chunk(content="!)"),
                ],
                style=weight_none,
            ),
            Text(
                content=[
                    Chunk(content="async", style=CellStyle(foreground=green_600)),
                    Chunk.space(),
                    Chunk(content="makes"),
                    Chunk.space(),
                    python_chunk,
                    Chunk.space(),
                    Chunk(content="faster", style=CellStyle(foreground=green_600, underline=True)),
                ],
                style=weight_none,
            ),
        ],
    )


@component
def processes_and_threads_in_memory() -> Div:
    w, h = 20, 20

    n_procs, set_n_procs = use_state(1)

    def on_key(event: KeyPressed) -> None:
        if event.key == "p":
            set_n_procs(lambda n: clamp(1, n + 1, 5))

    return Div(
        style=col | align_self_stretch,
        children=[
            Div(
                style=row | align_children_center,
                children=[
                    Text(
                        content=[
                            Chunk(content="Starting New"),
                            Chunk.newline(),
                            Chunk(content="Processes", style=CellStyle(foreground=lime_600)),
                        ],
                        style=pad_x_2 | weight_none | text_justify_center,
                    ),
                    Div(style=row, children=[random_walkers(w, h, False) for _ in range(n_procs)]),
                ],
            ),
            Div(
                style=row | align_children_center,
                children=[
                    Text(
                        content=[
                            Chunk(content="Starting New"),
                            Chunk.newline(),
                            Chunk(content="Threads", style=CellStyle(foreground=pink_600)),
                        ],
                        style=pad_x_2 | weight_none | text_justify_center,
                    ),
                    Div(
                        style=row,
                        children=[random_walkers(w, h, True)],
                    ),
                ],
            ),
        ],
        on_key=on_key,
    )


moves = [(x, y) for x, y in product((-1, 0, 1), repeat=2) if (x, y) != (0, 0)]


@component
def random_walkers(width: int, height: int, threads: bool) -> Text:
    colors, set_colors = use_state(random.sample(list(COLORS_BY_NAME.values()), k=1))
    walkers, set_walkers = use_state([(random.randrange(width), random.randrange(height)) for _ in range(len(colors))])

    def on_key(event: KeyPressed) -> None:
        if (event.key == "t" and threads) or (event.key == Key.ControlP and not threads):
            set_colors(lambda c: [*c, random.choice(list(COLORS_BY_NAME.values()))])
            set_walkers(lambda m: [*m, (random.randrange(width), random.randrange(height))])

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
        style=border_heavy | border_slate_400,
        on_key=on_key,
    )


asyncio.run(app(root))
