#!/usr/bin/env just --justfile

alias p := present
alias w := watch

present:
  python python_concurrency_and_parallelism/slides.py

watch:
  watchfiles 'python python_concurrency_and_parallelism/slides.py' python_concurrency_and_parallelism/
