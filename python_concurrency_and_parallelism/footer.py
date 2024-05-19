from asyncio import sleep
from datetime import datetime

from counterweight.components import component
from counterweight.elements import Chunk, Div, Text
from counterweight.events import KeyPressed
from counterweight.hooks import use_effect, use_rects, use_state
from counterweight.keys import Key
from counterweight.styles.utilities import *


@component
def footer(current_slide: int, total_slides: int) -> Div:
    current_time, set_current_time = use_state(datetime.now())
    show_time, set_show_time = use_state(True)
    show_size, set_show_size = use_state(False)

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.F1:
            set_show_time(lambda b: not b)
        elif event.key == Key.F2:
            set_show_size(lambda b: not b)

    async def tick() -> None:
        while True:
            await sleep(0.25)
            set_current_time(datetime.now())

    use_effect(tick, deps=())
    rects = use_rects()

    return Div(
        on_key=on_key,
        style=row
        | justify_children_space_between
        | weight_none
        | align_self_stretch
        | border_slate_400
        | border_light
        | border_top,
        children=list(
            filter(
                None,
                [
                    Text(
                        content=[Chunk(content="Principles of Concurrency & Parallelism in Python")],
                        style=text_slate_200,
                    ),
                    Text(content=f"{current_time:%Y-%m-%d %I:%M %p}", style=text_slate_200) if show_time else None,
                    Text(
                        content=f"{rects.content.width}",
                        style=text_green_200 if rects.content.width == 90 else text_red_200,
                    )
                    if show_size
                    else None,
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
        ),
    )
