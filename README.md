# circuit-creator

Get more instructions [here](https://github.com/ManimCommunity/manim)

## Installation

`sudo apt update`

`sudo apt upgrade`

`sudo apt install libpango1.0-dev pkg-config python3-dev`

`sudo apt-get install python3.8-dev` (change version)

`sudo apt install texlive-full`

`sudo apt install libcairo2-dev`

`sudo apt install ffmpeg`

`sudo apt-get install build-essential libgl1-mesa-dev`

`pip install -r requirements.txt`

THEN:

`pip install manim`

## Running

Add to .bashrc:

`export PYTHONPATH=$PYTHONPATH:insert_cwd`

In cwd Terminal:

`python -m manim [options] creation.py CircuitCreation`

Options:

* -p : __play__
* -ql : __quick load__
