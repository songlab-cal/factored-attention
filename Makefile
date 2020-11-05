SHELL := /bin/bash
.PHONY : build all update

all: update

build:
	git submodule update --init --recursive
	pip install -e ./mogwai
	pip install -r requirements.txt

update:
	git submodule sync --recursive
	git submodule update --recursive
	pip install -e ./mogwai