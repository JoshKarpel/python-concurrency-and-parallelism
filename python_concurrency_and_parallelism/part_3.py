from textwrap import dedent

from counterweight.components import component
from counterweight.elements import Chunk, Div, Text
from counterweight.styles import CellStyle
from counterweight.styles.utilities import *

from python_concurrency_and_parallelism.utils import drop_shadow


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
          (i.e., some other system accessed over a network connection)
        """
    )


@component
def scenario_3() -> Div:
    return scenario(
        """\
        - You are building an HTTP service whose primary job is
          to forward requests to other services over the network (i.e., a proxy)

        - Your proxy needs to be able to handle a very
          large number of simultaneous incoming requests

        - The services you are proxying to are very slow to respond
          (compared to how long it takes you to figure out which one to forward to)
        """
    )


PART_3 = [
    part_3,
    scenario_1,
    scenario_2,
    scenario_3,
]
