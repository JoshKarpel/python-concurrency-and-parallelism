import asyncio
import inspect
import random
import sys
from asyncio import sleep
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from itertools import chain, product, repeat
from math import ceil, floor
from sys import getswitchinterval
from textwrap import dedent
from time import time_ns
from typing import Type
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from counterweight.app import app
from counterweight.components import component
from counterweight.controls import Quit, Screenshot
from counterweight.elements import Chunk, Div, Text
from counterweight.events import KeyPressed
from counterweight.hooks import use_effect, use_state
from counterweight.keys import Key
from counterweight.styles.styles import COLORS_BY_NAME
from counterweight.styles.utilities import *
from more_itertools import flatten, intersperse, take
from pygments.lexers import get_lexer_by_name
from pygments.styles import get_style_by_name
from structlog import get_logger

from python_concurrency_and_parallelism.utils import black, canvas, clamp

logger = get_logger()

full_block = "█"

python_blue = Color.from_hex("#4B8BBE")
python_chunk = Chunk(content="Python", style=CellStyle(foreground=python_blue))


@component
def root() -> Div:
    current_slide, set_current_slide = use_state(0)

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Right:
            set_current_slide(lambda n: clamp(0, n + 1, len(SLIDES) - 1))
        elif event.key == Key.Left:
            set_current_slide(lambda n: clamp(0, n - 1, len(SLIDES) - 1))

    return Div(
        style=col,
        children=[
            Div(
                style=col | align_self_stretch,
                children=[SLIDES[current_slide]()],
            ),
            footer(current_slide=current_slide + 1, total_slides=len(SLIDES)),
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
                content=[Chunk(content="Concurrency & Parallelism in Python")],
                style=text_slate_200,
            ),
            # Text(content=f"{current_time:%Y-%m-%d %I:%M %p}", style=text_slate_200),
            Text(
                content=[
                    Chunk(content=f"{current_slide}"),
                    Chunk(content=" / "),
                    Chunk(content=f"{total_slides}"),
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
def part_1() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            drop_shadow(
                Text(
                    content=[
                        Chunk(content="Part #1", style=CellStyle(underline=True)),
                        Chunk.newline(),
                        Chunk.newline(),
                        Chunk(content="How Computers Work"),
                    ],
                    style=weight_none | border_heavy | border_gray_200 | text_justify_center | pad_x_1,
                ),
            ),
        ],
    )


@component
def part_2() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            drop_shadow(
                Text(
                    content=[
                        Chunk(content="Part #2", style=CellStyle(underline=True)),
                        Chunk.newline(),
                        Chunk.newline(),
                        Chunk(content="How Python Works"),
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
        Chunk(content="A Python process can"),
        Chunk.space(),
        Chunk(content="never", style=CellStyle(foreground=red_600, underline=True)),
        Chunk.space(),
        Chunk(content="execute more Python bytecode"),
        Chunk.newline(),
        Chunk(content="per unit time than a single Python thread can"),
    ]
    if revealed:
        content += [
            Chunk.newline(),
            Chunk.newline(),
            Chunk(content="* For CPython with a GIL; see PEP 703", style=CellStyle(foreground=amber_600)),
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
def computers() -> Div:
    arrow_shift, set_arrow_shift = use_state(0)

    arrow = Chunk(content=" " * arrow_shift + "↑", style=CellStyle(foreground=python_blue))

    async def tick() -> None:
        while True:
            await sleep(0.25)
            set_arrow_shift(lambda n: (n + 1) % 20)

    use_effect(tick, deps=())

    half_and_half_div_style = col | align_children_center | gap_children_1
    color_bar_div_style = col | border_heavy | border_gray_400 | pad_x_1

    c = Div(
        style=half_and_half_div_style,
        children=[
            Text(
                content=[
                    Chunk(content="Modern computers are usually"),
                    Chunk.space(),
                    Chunk(content="concurrent", style=CellStyle(foreground=green_600)),
                    Chunk.space(),
                    Chunk(content="and", style=CellStyle(underline=True)),
                    Chunk.space(),
                    Chunk(content="parallel", style=CellStyle(foreground=green_600)),
                ],
                style=weight_none,
            ),
            Div(
                style=color_bar_div_style,
                children=[
                    Text(
                        content=colored_bar(
                            (9, palette[0]),
                            (7, palette[1]),
                            (4, palette[4]),
                        ),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar(
                            (7, palette[1]),
                            (5, palette[5]),
                            (8, palette[3]),
                        ),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar(
                            (3, palette[5]),
                            (11, palette[2]),
                            (6, palette[0]),
                        ),
                        style=weight_none,
                    ),
                    Text(
                        content=[arrow],
                        style=weight_none,
                    ),
                    Text(
                        content=colored_bar(
                            (3, palette[6]),
                            (12, palette[4]),
                            (3, palette[2]),
                            (2, palette[5]),
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

    return Div(
        style=row | align_self_stretch | align_children_center | justify_children_space_around,
        children=[c],
    )


def task(name: str, n: int) -> Generator[tuple[str, int], None, None]:
    for idx in range(n):
        # Make some incremental progress on the task here
        yield name, idx + 1


def cooperative_concurrency() -> None:
    tasks = deque(
        [
            task("A", 3),
            task("B", 4),
            task("C", 2),
        ]
    )
    while tasks:
        task_to_run = tasks.popleft()
        try:
            name, count = next(task_to_run)
            print(f"{name} {count}")
            tasks.append(task_to_run)
        except StopIteration:
            pass
    print("Done")


def cooperative_concurrency_real() -> Iterator[tuple[int, str]]:
    ready = deque(
        [
            task("A", 3),
            task("B", 4),
            task("C", 2),
        ]
    )
    while ready:
        task_to_run = ready.popleft()
        try:
            name, idx = next(task_to_run)
            yield f"{name}  {idx}"
            ready.append(task_to_run)
        except StopIteration:
            pass
    yield "Done"


@component
def cooperative_concurrency_example() -> Div:
    outputs, set_outputs = use_state([])

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Space:
            set_outputs(list(cooperative_concurrency_real()))

    return Div(
        on_key=on_key,
        style=row | align_self_stretch | align_children_center | justify_children_center | gap_children_1,
        children=[
            make_code_example(task, cooperative_concurrency),
            Text(
                style=weight_none | pad_x_1 | (border_heavy if outputs else None),
                content=list(intersperse(Chunk.newline(), [Chunk(content=output) for output in outputs])),
            ),
        ],
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
def processes_and_threads() -> Div:
    w, h = 20, 20

    parts = ("Isolated Memory", "Shared Memory")
    justify_width = max(len(p) for p in parts)
    parts = [p.center(justify_width) for p in parts]

    return Div(
        style=col | align_self_stretch | pad_x_2,
        children=[
            Div(
                style=row | align_children_center | align_self_stretch,
                children=[
                    Text(
                        content=[
                            Chunk(content="Processes", style=CellStyle(foreground=lime_600)),
                            Chunk.newline(),
                            Chunk(content=parts[0]),
                        ],
                        style=pad_x_2 | weight_none | text_justify_center,
                    ),
                    Div(
                        style=row | justify_children_center,
                        children=[random_walkers(w, h, 1, first=idx == 0) for idx in range(3)],
                    ),
                ],
            ),
            Div(
                style=row | align_children_center | align_self_stretch,
                children=[
                    Text(
                        content=[
                            Chunk(content="Threads", style=CellStyle(foreground=pink_600)),
                            Chunk.newline(),
                            Chunk(content=parts[1]),
                        ],
                        style=pad_x_2 | weight_none | text_justify_center,
                    ),
                    Div(style=row | justify_children_center, children=[random_walkers(w, h, 6, first=True)]),
                ],
            ),
        ],
    )


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


@component
def tools() -> Div:
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


def track_activity(start_time: float, run_for: float, bucket_size: float) -> list[int]:
    """
    The tracker is a list of buckets representing time slices.
    Each bucket will be the count of how many times
    the below loop ran during that time slice.
    The bucket count is thus roughly proportional to
    the amount of Python bytecode executed in that time slice.
    """
    tracker = [0 for _ in range(ceil(run_for / bucket_size))]

    while True:
        time_since_start = time_ns() - start_time

        if time_since_start >= run_for:
            break

        bucket_index = floor(time_since_start / bucket_size)
        tracker[bucket_index] += 1

    return tracker


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


def make_code_example(*fns: Callable[[...], object]) -> Div:
    lexer = get_lexer_by_name("python")

    chunks = []
    for fn in fns:
        for token, text in lexer.get_tokens(inspect.getsource(fn)):
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


@component
def activity_tracking_example() -> Div:
    return make_code_example(track_activity)


@component
def what_the_gil_actually_does() -> Div:
    bucket_size = (getswitchinterval() / 5) * 1e9
    buckets = 60
    offset = 300
    total_buckets = offset + buckets + (offset // 2)
    run_for = total_buckets * bucket_size
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
                    start_time=start_time,
                    run_for=run_for,
                    bucket_size=bucket_size,
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

    process_bars = make_activity_bars(buckets, offset, process_results)
    thread_bars = make_activity_bars(buckets, offset, thread_results)

    processes = Div(
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
    threads = Div(
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

    return Div(
        style=col | align_self_stretch | align_children_center | justify_children_space_around,
        children=[
            processes,
            threads,
            Text(
                style=weight_none | text_justify_center,
                content=f"Bucket Size = {bucket_size / 1e6:.3f} milliseconds\nGIL Switch Interval = {sys.getswitchinterval()*1e3=:.3f} milliseconds",
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
                    *((1, black.blend(palette[n], t / biggest_count)) for t in tracker[offset : offset + buckets])
                ),
                Chunk.space(),
                Chunk(content=f"{sum(tracker[offset:offset + buckets]):>6}"),
            ],
            style=weight_none,
        )
        for n, tracker in enumerate(thread_results)
    ] + [time_arrow(10)]


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
                        Chunk(content="The event loop runs on a single thread and is"),
                        Chunk.newline(),
                        Chunk(content="exclusively", style=CellStyle(foreground=red_600, underline=True)),
                        Chunk.space(),
                        Chunk(content="cooperatively concurrent"),
                    ],
                    style=weight_none | border_heavy | border_gray_200 | text_justify_center | pad_x_1,
                ),
            ),
        ],
    )


@component
def blocking_the_event_loop() -> Div:
    reveals, set_reveals = use_state(0)

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Space:
            set_reveals(lambda n: n + 1)

    half_and_half_div_style = col | align_children_center | gap_children_1

    return Div(
        style=row | align_self_stretch | justify_children_space_around | pad_top_3,
        on_key=on_key,
        children=[
            Div(
                style=half_and_half_div_style,
                children=[
                    Text(
                        content=[
                            Chunk(content="things that ", style=CellStyle(underline=True)),
                            Chunk(content="do not", style=CellStyle(foreground=green_600, underline=True)),
                            Chunk(content=" block the event loop", style=CellStyle(underline=True)),
                        ],
                        style=weight_none,
                    ),
                    Text(
                        content=[
                            Chunk(content="await", style=CellStyle(foreground=python_blue)),
                            Chunk(content="ing something that isn't ready yet"),
                        ],
                        style=weight_none,
                    )
                    if reveals >= 1
                    else Text(content=""),
                    Text(
                        content=[
                            Chunk(content="equivalent syntax sugar like "),
                            Chunk(content="async for", style=CellStyle(foreground=python_blue)),
                        ],
                        style=weight_none,
                    )
                    if reveals >= 2
                    else Text(content=""),
                ],
            ),
            Div(
                style=half_and_half_div_style,
                children=[
                    Text(
                        content=[
                            Chunk(content="things that ", style=CellStyle(underline=True)),
                            Chunk(content="do", style=CellStyle(foreground=red_600, underline=True)),
                            Chunk(content=" block the event loop", style=CellStyle(underline=True)),
                        ],
                        style=weight_none,
                    ),
                    Text(
                        content="everything else",
                        style=weight_none,
                    )
                    if reveals >= 3
                    else Text(content=""),
                    Text(
                        content="yes, even when you release the GIL",
                        style=weight_none,
                    )
                    if reveals >= 3
                    else Text(content=""),
                ],
            ),
        ],
    )


@component
def part_3() -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            drop_shadow(
                Text(
                    content=[
                        Chunk(content="Part #3", style=CellStyle(underline=True)),
                        Chunk.newline(),
                        Chunk.newline(),
                        Chunk(content="Example Scenarios"),
                    ],
                    style=weight_none | border_heavy | border_gray_200 | text_justify_center | pad_x_1,
                ),
            ),
        ],
    )


def scenario(text: str) -> Div:
    return Div(
        style=col | align_self_center | align_children_center | justify_children_center | gap_children_2,
        children=[
            Text(
                content=dedent(text),
                style=weight_none | pad_x_1,
            ),
        ],
    )


@component
def scenario_1() -> Div:
    return scenario(
        """\
        - You have a large list of numpy arrays stored in files on disk

        - You need to do the same long series of operations on each array independently

        - Once all the operations are done,
          you need to sum the resulting arrays to get a single result array
        """
    )


@component
def scenario_2() -> Div:
    return scenario(
        """\
        - You are building a web crawler,
          which will read a large number of web pages via HTTP requests

        - Once you get the contents of each page,
          you will run a very slow algorithm written in pure Python on it

        - The results of that slow operation will be stored in a database
          (i.e., some other system)
        """
    )


@component
def scenario_3() -> Div:
    return scenario(
        """\
        - You are building an HTTP service whose primary job is
          to forward requests to other services (i.e., a proxy)

        - Your proxy needs to be able to handle a very
          large number of simultaneous incoming requests

        - The services you are proxying to are very slow to respond
          compared to how long it takes you to figure out which one to forward to
        """
    )


SLIDES = [
    title,
    rule_0,
    you_may_have_heard,
    part_1,
    definitions,
    computers,
    processes_and_threads,
    cooperative_concurrency_example,
    tools,
    part_2,
    activity_tracking_example,
    what_the_gil_actually_does,
    rule_1,
    # TODO: introduce python asyncio here
    rule_2,
    blocking_the_event_loop,
    # TODO: connect python mechanisms to the earlier tools slide
    part_3,
    scenario_1,
    scenario_2,
    scenario_3,
]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        asyncio.run(app(root))
    else:
        html = Element("html")
        et = ElementTree(element=html)

        style = SubElement(html, "style")
        style.text = dedent(
            """\
            .wrapper {
              display: flex;
              flex-direction: column;
              gap: 2em;
            }

            .slide {
              display: flex;
              flex-direction: row;
              align-items: center;
            }

            svg {
              display: block;
              margin: auto;
            }
            """
        )

        body = SubElement(html, "body")
        wrapper = SubElement(body, "div", attrib={"class": "wrapper"})

        def aggregator(svg: ElementTree) -> None:
            slide = SubElement(wrapper, "div", attrib={"class": "slide"})
            slide.append(svg.getroot())

        asyncio.run(
            app(
                root,
                dimensions=(100, 30),
                headless=True,
                autopilot=chain(
                    flatten(
                        take(
                            len(SLIDES),
                            zip(
                                repeat(Screenshot(handler=aggregator)),
                                repeat(KeyPressed(key=Key.Right)),
                            ),
                        ),
                    ),
                    (Quit(),),
                ),
            )
        )

        indent(et, space="  ")

        et.write("slides.html", encoding="unicode")

        print("done")
