#!/usr/bin/env just --justfile

alias p := present
alias w := watch

present:
  python python_concurrency_and_parallelism/slides.py

watch:
  watchfiles 'python python_concurrency_and_parallelism/slides.py' python_concurrency_and_parallelism/ ../counterweight/counterweight/

html:
  python python_concurrency_and_parallelism/slides.py html

watch-html:
  watchfiles 'python python_concurrency_and_parallelism/slides.py html' python_concurrency_and_parallelism/ ../counterweight/counterweight/
