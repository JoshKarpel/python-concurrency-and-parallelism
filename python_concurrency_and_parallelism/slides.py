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


@component
def root() -> Div:
    slides = [
        processes_and_threads_in_memory,
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
                        content="Python Concurrency & Parallelism",
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
