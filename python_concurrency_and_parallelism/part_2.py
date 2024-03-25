import subprocess
import sys
from asyncio import gather, run, sleep
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from math import ceil, floor
from sys import getswitchinterval
from time import time_ns
from typing import Type

from counterweight.components import component
from counterweight.elements import Chunk, Div, Text
from counterweight.events import KeyPressed
from counterweight.hooks import use_state
from counterweight.keys import Key
from counterweight.styles import CellStyle
from counterweight.styles.utilities import *
from more_itertools import intersperse
from structlog import get_logger

from python_concurrency_and_parallelism.utils import drop_shadow, make_activity_bars, make_code_example, python_blue

logger = get_logger()


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
            set_revealed(not revealed)

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
                    if reveals >= 4
                    else Text(content=""),
                ],
            ),
        ],
    )


async def async_task(name: str, n: int) -> None:
    for idx in range(n):
        # Make some incremental progress on the task here,
        # and then wait for something to happen!
        await sleep(0.01)
        print(f"{name}  {idx + 1}")


async def async_cooperative_concurrency() -> None:
    await gather(
        async_task("A", 3),
        async_task("B", 4),
        async_task("C", 2),
    )

    print("Done")


@component
def async_cooperative_concurrency_example() -> Div:
    outputs, set_outputs = use_state([])

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Space:
            logger.debug("hi")
            set_outputs(
                subprocess.run(["python", __file__, "good"], capture_output=True, text=True).stdout.strip().splitlines()
            )

    return Div(
        on_key=on_key,
        style=row | align_self_stretch | align_children_center | justify_children_center | gap_children_1,
        children=[
            make_code_example(async_task, async_cooperative_concurrency),
            Text(
                style=weight_none | pad_x_1 | (border_heavy if outputs else None),
                content=list(intersperse(Chunk.newline(), [Chunk(content=output) for output in outputs])),
            ),
        ],
    )


async def bad_async_task(name: str, n: int) -> None:
    for idx in range(n):
        # Make some incremental progress on the task here,
        # and then keep doing more work!
        print(f"{name}  {idx + 1}")


async def bad_async_cooperative_concurrency() -> None:
    await gather(
        bad_async_task("A", 3),
        bad_async_task("B", 4),
        bad_async_task("C", 2),
    )

    print("Done")


@component
def bad_async_cooperative_concurrency_example() -> Div:
    outputs, set_outputs = use_state([])

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Space:
            logger.debug("hi")
            set_outputs(
                subprocess.run(["python", __file__, "bad"], capture_output=True, text=True).stdout.strip().splitlines()
            )

    return Div(
        on_key=on_key,
        style=row | align_self_stretch | align_children_center | justify_children_center | gap_children_1,
        children=[
            make_code_example(bad_async_task, bad_async_cooperative_concurrency),
            Text(
                style=weight_none | pad_x_1 | (border_heavy if outputs else None),
                content=list(intersperse(Chunk.newline(), [Chunk(content=output) for output in outputs])),
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


# via https://developer.nvidia.com/blog/building-a-machine-learning-microservice-with-fastapi/
terrible = r"""
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleCarTransactionInputData) -> Any:
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(inputs=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results
"""


def something_that_make_josh_furious() -> Div:
    return Div(
        style=col | align_self_center | align_children_center,
        children=[
            make_code_example(terrible),
            Text(
                style=weight_none,
                content="via https://developer.nvidia.com/blog/building-a-machine-learning-microservice-with-fastapi/",
            ),
        ],
    )


PART_2 = [
    part_2,
    activity_tracking_example,
    what_the_gil_actually_does,
    rule_1,
    async_cooperative_concurrency_example,
    bad_async_cooperative_concurrency_example,
    rule_2,
    blocking_the_event_loop,
    something_that_make_josh_furious,
]

if __name__ == "__main__":
    run(async_cooperative_concurrency() if sys.argv[1] == "good" else bad_async_cooperative_concurrency())
