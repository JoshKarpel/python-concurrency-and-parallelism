import asyncio
import random
from asyncio import sleep
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product
from math import floor
from sys import getswitchinterval
from time import time_ns
from typing import Type

from reprisal.app import app
from reprisal.components import Chunk, Div, Text, component
from reprisal.events import KeyPressed
from reprisal.hooks import use_effect, use_state
from reprisal.keys import Key
from reprisal.styles.styles import COLORS_BY_NAME
from reprisal.styles.utilities import *
from structlog import get_logger

from python_concurrency_and_parallelism.utils import black, canvas, clamp

logger = get_logger()

full_block = "█"

python_blue = Color.from_hex("#4B8BBE")
python_chunk = Chunk(content="Python", style=CellStyle(foreground=python_blue))


@component
def root() -> Div:
    slides = [
        title,
        rule_0,
        you_may_have_heard,
        definitions,
        processes_and_threads_in_memory,
        memory_sharing_and_multitasking,
        what_the_gil_actually_does,
        rule_1,
        rule_2,
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
    # current_time, set_current_time = use_state(datetime.now())
    #
    # async def tick() -> None:
    #     while True:
    #         await sleep(1 / 60)
    #         set_current_time(datetime.now())
    #
    # use_effect(tick, deps=())

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
            # Text(content=f"{current_time:%Y-%m-%d %I:%M %p}", style=text_slate_200),
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


def drop_shadow(*c: Text | Div) -> Div:
    return Div(
        style=border_lightshade | border_bottom_right | border_contract_1 | border_gray_400 | weight_none,
        children=list(c),
    )


@component
def title() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            drop_shadow(
                Text(
                    content="Concurrency & Parallelism in Python",
                    style=weight_none | border_heavy | border_gray_200 | pad_x_1,
                ),
            ),
            Text(content="Josh Karpel", style=weight_none),
            Text(content="MadPy, May 2024", style=weight_none | text_gray_400),
        ],
    )


@component
def rule_0() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            drop_shadow(
                Text(
                    content=[
                        Chunk(content="Rule #0", style=CellStyle(underline=True)),
                        Chunk.newline(),
                        Chunk.newline(),
                        Chunk(content="The only thing that matters is the performance of your actual system"),
                    ],
                    style=weight_none | border_heavy | border_gray_200 | text_justify_center | pad_x_1,
                ),
            ),
        ],
    )


@component
def rule_1() -> Div:
    revealed, set_revealed = use_state(False)

    content = [
        Chunk(content="Rule #1", style=CellStyle(underline=True)),
        Chunk.newline(),
        Chunk.newline(),
        Chunk(content="A Python process can never execute more Python bytecode"),
        Chunk.newline(),
        Chunk(content="per unit time than a single Python thread can"),
    ]
    if revealed:
        content += [
            Chunk.newline(),
            Chunk.newline(),
            Chunk(content="* If it has a GIL; see PEP 703", style=CellStyle(foreground=amber_600)),
        ]

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Space:
            set_revealed(toggle)

    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        on_key=on_key,
        children=[
            drop_shadow(
                Text(
                    content=content,
                    style=weight_none | border_heavy | border_gray_200 | text_justify_center | pad_x_1,
                ),
            ),
        ],
    )


@component
def rule_2() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            drop_shadow(
                Text(
                    content=[
                        Chunk(content="Rule #2", style=CellStyle(underline=True)),
                        Chunk.newline(),
                        Chunk.newline(),
                        Chunk(content="Don't block the event loop!"),
                    ],
                    style=weight_none | border_heavy | border_gray_200 | text_justify_center | pad_x_1,
                ),
            ),
        ],
    )


palette = [Color.from_hex(c) for c in ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d")]


def colored_bar(*blocks: tuple[int, Color]) -> list[Chunk]:
    return [Chunk(content=full_block * n, style=CellStyle(foreground=c)) for n, c in blocks]


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

    half_and_half_div_style = col | align_children_center | gap_children_1
    color_bar_div_style = col | border_heavy | border_gray_400 | pad_x_1

    concurrency = Div(
        style=half_and_half_div_style,
        children=[
            Text(
                content=[Chunk(content="Concurrency", style=CellStyle(underline=True))],
                style=weight_none,
            ),
            Text(
                content=[
                    Chunk(content="Tasks are"),
                    Chunk.space(),
                    Chunk(content="interleaved", style=CellStyle(foreground=green_600)),
                ],
                style=weight_none,
            ),
            Div(
                style=color_bar_div_style,
                children=[
                    Text(
                        content=colored_bar(
                            (3, palette[0]),
                            (2, palette[1]),
                            (1, palette[2]),
                            (5, palette[0]),
                            (2, palette[2]),
                            (3, palette[3]),
                            (1, palette[1]),
                            (3, palette[0]),
                        ),
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
                style=weight_none,
            ),
            Text(
                content=[
                    Chunk(content="Tasks are"),
                    Chunk.space(),
                    Chunk(content="simultaneous", style=CellStyle(foreground=green_600)),
                ],
                style=weight_none,
            ),
            Div(
                style=color_bar_div_style,
                children=[
                    Text(
                        content=colored_bar((20, palette[0])),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar((11, palette[1]), (9, palette[5])),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar((14, palette[2]), (6, palette[3])),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar((20, palette[6])),
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
            set_n_procs(lambda n: clamp(1, n + 1, 3))

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


@component
def memory_sharing_and_multitasking() -> Div:
    revealed, set_revealed = use_state(False)

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Space:
            set_revealed(toggle)

    table = [
        ["Scheduling", "+", "Memory", "=", "Tool"],
        ["Preemptive", "+", "Shared", "=", "Threads"],
        ["Preemptive", "+", "Isolated", "=", "Processes"],
        ["Cooperative", "+", "Shared", "=", "Async"],
    ]
    revealable = ["Cooperative", "+", "Isolated", "=", "Queues / Actors"]
    if revealed:
        table.append(revealable)
    else:
        table.append([" " * len(s) for s in revealable])

    return Div(
        style=col | align_self_stretch | align_children_center | justify_children_center,
        on_key=on_key,
        children=[
            Div(
                style=row | align_self_stretch | align_children_center | justify_children_center | weight_none,
                children=[
                    Div(
                        style=col | weight_none,
                        children=[
                            Text(
                                style=weight_none
                                | pad_x_1
                                | (
                                    border_heavy | border_slate_400
                                    if r == 0 and c != 1 and c != 3
                                    else pad_y_1 | pad_x_2
                                )
                                | (text_amber_500 if r == 4 else None),
                                content=table[r][c],
                            )
                            for r in range(len(table))
                        ],
                    )
                    for c in range(len(table[0]))
                ],
            )
        ],
    )


def track_activity(start_time: int, stop_time: float, offset: int, buckets: int, bucket_size_ns: float) -> list[int]:
    tracker = [0 for _ in range(offset + buckets + offset + 10)]

    while True:
        current_time = time_ns()
        delta_ns = current_time - start_time

        if delta_ns > stop_time:
            break

        bucket = floor(delta_ns / bucket_size_ns)
        tracker[bucket] += 1

    return tracker


@component
def what_the_gil_actually_does() -> Div:
    bucket_size_ns = (getswitchinterval() / 5) * 1e9
    buckets = 60
    offset = 500
    total_buckets = offset + buckets + (offset // 2)
    stop_time = total_buckets * bucket_size_ns
    zeros = [0] * total_buckets

    N = 4

    thread_results, set_thread_results = use_state([zeros] * N)
    process_results, set_process_results = use_state([zeros] * N)

    def run_experiment(executor_type: Type[ThreadPoolExecutor] | Type[ProcessPoolExecutor]) -> list[list[int]]:
        with executor_type(max_workers=N) as executor:
            start_time = time_ns()
            trackers = [
                executor.submit(
                    track_activity,
                    start_time,
                    stop_time,
                    offset,
                    buckets,
                    bucket_size_ns,
                )
                for _ in range(N)
            ]

        return [tracker.result() for tracker in trackers]

    def on_key(event: KeyPressed) -> None:
        if event.key == "t":
            set_thread_results(run_experiment(ThreadPoolExecutor))
        elif event.key == "p":
            set_process_results(run_experiment(ProcessPoolExecutor))

    half_and_half_div_style = col | align_children_center | gap_children_1
    color_bar_div_style = col | border_heavy | border_gray_400

    thread_bars = make_activity_bars(buckets, offset, thread_results)
    process_bars = make_activity_bars(buckets, offset, process_results)

    concurrency = Div(
        style=half_and_half_div_style,
        on_key=on_key,
        children=[
            Text(
                content=[Chunk(content=f"{N} Threads in 1 Process")],
                style=weight_none,
            ),
            Div(
                style=color_bar_div_style,
                children=thread_bars,
            ),
        ],
    )
    parallelism = Div(
        style=half_and_half_div_style,
        children=[
            Text(
                content=[Chunk(content=f"{N} Processes, 1 Thread Each")],
                style=weight_none,
            ),
            Div(
                style=color_bar_div_style,
                children=process_bars,
            ),
        ],
    )

    return Div(
        style=col | align_self_stretch | align_children_center | justify_children_space_around,
        children=[concurrency, parallelism],
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
                Chunk(content=f"  T{n} "),
                *colored_bar(
                    *((1, black.blend(palette[n], t / biggest_count)) for t in tracker[offset : offset + buckets])
                ),
                Chunk.space(),
                Chunk(content=f"{sum(tracker[offset:offset + buckets]):>6}"),
            ],
            style=weight_none,
        )
        for n, tracker in enumerate(thread_results)
    ] + [time_arrow(10)]


asyncio.run(app(root))
