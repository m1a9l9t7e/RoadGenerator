# circuit-creator

## Installation

`sudo apt update`

`sudo apt upgrade`

`sudo apt install libpango1.0-dev pkg-config python3-dev`

`sudo apt install texlive-full`

`sudo apt install libcairo2-dev`

`sudo apt install ffmpeg`

`sudo apt-get install build-essential libgl1-mesa-dev`

`pip install manimpango`

## Running

Add to .bashrc:

`export PYTHONPATH=$PYTHONPATH:insert_cwd`

In cwd Terminal:

`manimgl [options] scene.py CircuitCreation`

Options:

* -p : __play__
* -ql : __quick load__

## Instructions for cloning Manim from Github (needed for moving camera)

Get Code from [here](https://github.com/ManimCommunity/manim)

`pip install manim`

Execute with:

`python -m manim [options] scene.py CircuitCreation`
