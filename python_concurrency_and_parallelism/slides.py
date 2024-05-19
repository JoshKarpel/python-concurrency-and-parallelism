#!/usr/bin/env python

import asyncio
import sys
from itertools import chain, repeat
from pathlib import Path
from tempfile import gettempdir
from textwrap import dedent
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from counterweight.app import app
from counterweight.components import component
from counterweight.controls import Quit, Screenshot
from counterweight.elements import Div
from counterweight.events import KeyPressed
from counterweight.hooks import use_state
from counterweight.keys import Key
from counterweight.styles.utilities import *
from more_itertools import flatten, take
from structlog import get_logger

from python_concurrency_and_parallelism.footer import footer
from python_concurrency_and_parallelism.part_0 import PART_0
from python_concurrency_and_parallelism.part_1 import PART_1
from python_concurrency_and_parallelism.part_2 import PART_2
from python_concurrency_and_parallelism.part_3 import PART_3
from python_concurrency_and_parallelism.utils import clamp

logger = get_logger()

SLIDE_IDX_FILE = Path(gettempdir()) / "python-concurrency-and-parallelism-slide-idx"


def load_current_slide() -> int:
    SLIDE_IDX_FILE.touch(exist_ok=True)
    return clamp(0, int(SLIDE_IDX_FILE.read_text().strip() or "0"), len(SLIDES) - 1)


def write_current_slide(current_slide: int) -> int:
    SLIDE_IDX_FILE.write_text(str(current_slide))
    return current_slide


@component
def root() -> Div:
    current_slide, set_current_slide = use_state(load_current_slide)

    def on_key(event: KeyPressed) -> None:
        if event.key == Key.Right:
            set_current_slide(lambda n: write_current_slide(clamp(0, n + 1, len(SLIDES) - 1)))
        elif event.key == Key.Left:
            set_current_slide(lambda n: write_current_slide(clamp(0, n - 1, len(SLIDES) - 1)))

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


SLIDES = [
    *PART_0,
    *PART_1,
    *PART_2,
    *PART_3,
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

        def screenshot_aggregator(svg: ElementTree) -> None:
            slide = SubElement(wrapper, "div", attrib={"class": "slide"})
            slide.append(svg.getroot())

        write_current_slide(0)
        asyncio.run(
            app(
                root,
                dimensions=(90, 30),
                headless=True,
                autopilot=chain(
                    flatten(
                        take(
                            len(SLIDES),
                            zip(
                                repeat(Screenshot(handler=screenshot_aggregator)),
                                repeat(KeyPressed(key=Key.Right)),
                            ),
                        ),
                    ),
                    (Quit(),),
                ),
            )
        )

        indent(et, space="  ")

        et.write("slides.html", encoding="utf-8")
