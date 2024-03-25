from asyncio import sleep
from datetime import datetime

from counterweight.components import component
from counterweight.elements import Chunk, Div, Text
from counterweight.hooks import use_effect, use_state
from counterweight.styles.utilities import *


@component
def footer(current_slide: int, total_slides: int) -> Div:
    current_time, set_current_time = use_state(datetime.now())

    async def tick() -> None:
        while True:
            await sleep(0.25)
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
                content=[Chunk(content="Principles of Concurrency & Parallelism in Python")],
                style=text_slate_200,
            ),
            Text(content=f"{current_time:%Y-%m-%d %I:%M %p}", style=text_slate_200),
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
