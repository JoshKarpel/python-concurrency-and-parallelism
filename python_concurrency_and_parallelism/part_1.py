from asyncio import sleep
from collections import deque
from typing import Iterator

from counterweight.components import component
from counterweight.elements import Chunk, Div, Text
from counterweight.events import KeyPressed
from counterweight.hooks import use_effect, use_state
from counterweight.keys import Key
from counterweight.styles.utilities import *
from more_itertools import intersperse

from python_concurrency_and_parallelism.utils import (
    colored_bar,
    drop_shadow,
    make_code_example,
    palette,
    python_blue,
    random_walkers,
    time_arrow,
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
def the_goal() -> Div:
    half_and_half_div_style = col | align_children_center | gap_children_1

    c = Div(
        style=half_and_half_div_style,
        children=[
            Text(
                content=[
                    Chunk(content="Computers are"),
                    Chunk.space(),
                    Chunk(content="so fast", style=CellStyle(foreground=green_600)),
                    Chunk.space(),
                    Chunk(content="that they often need to"),
                    Chunk.space(),
                    Chunk(content="wait", style=CellStyle(foreground=red_600)),
                    Chunk.space(),
                    Chunk(content="for things to happen"),
                ],
                style=weight_none | text_justify_center,
            ),
        ],
    )

    return Div(
        style=row | align_self_stretch | align_children_center | justify_children_space_around,
        children=[c],
    )


@component
def the_table() -> Div:
    return Div(
        style=col | align_self_stretch | align_children_center | justify_children_center | gap_children_2,
        children=[
            Text(
                style=weight_none | border_heavy | pad_x_1,
                content="""\
Action                        Time       Relative Time
---------------------------   ---------  -------------
1 CPU cycle                   0.3 ns     1 s
Level 1 cache access          0.9 ns     3 s
Level 2 cache access          2.8 ns     9 s
Level 3 cache access          12.9 ns    43 s
Main memory access            120 ns     6 min
Solid-state disk I/O          50-150 μs  2-6 days
Rotational disk I/O           1-10 ms    1-12 months
Internet: SF to NYC           40 ms      4 years
Internet: SF to UK            81 ms      8 years
Internet: SF to Australia     183 ms     19 years
OS virtualization reboot      4 s        423 years
SCSI command time-out         30 s       3000 years
Physical system reboot        5 m        32 millenia""",
            ),
            Text(
                style=weight_none,
                content="Adapted from Systems Performance: Enterprise and the Cloud, by Brendan Gregg",
            ),
        ],
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


def task(name: str, n: int) -> Iterator[tuple[str, int]]:
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
        style=row | align_self_stretch | align_children_center | justify_children_center | gap_children_1 | pad_x_2,
        children=[
            Text(
                style=weight_none | text_justify_center | pad_left_2,
                content=[
                    Chunk(content="Coroutines", style=CellStyle(foreground=python_blue)),
                    Chunk.newline(),
                    Chunk(content="Shared Memory"),
                ],
            ),
            make_code_example(task, cooperative_concurrency),
            Text(
                style=weight_none | (border_heavy if outputs else pad_x_3),
                content=list(intersperse(Chunk.newline(), [Chunk(content=output) for output in outputs])),
            ),
        ],
    )


@component
def tools() -> Div:
    revealed, set_revealed = use_state(False)

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Space:
            set_revealed(not revealed)

    table = [
        ["Scheduling", "+", "Memory", "=", "Tool"],
        ["Preemptive", "+", "Shared", "=", "Threads"],
        ["Preemptive", "+", "Isolated", "=", "Processes"],
        ["Cooperative", "+", "Shared", "=", "Coroutines / Async"],
    ]
    revealable = ["Cooperative", "+", "Isolated", "=", "Async Actors?"]
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


PART_1 = [
    part_1,
    the_goal,
    the_table,
    definitions,
    computers,
    processes_and_threads,
    cooperative_concurrency_example,
    tools,
]
