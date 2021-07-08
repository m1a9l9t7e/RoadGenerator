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

## CPLEX

If you want to use CPlEX (CPLEX ILOG must be installed):

`sudo python /opt/ibm/ILOG/CPLEX_Studio201/python/setup.py install`

add the following to .bashrc (change python version and cplex version):

`export CPLEX_HOME="/opt/ibm/ILOG/CPLEX_Studio201/cplex"`

`export CPO_HOME="/opt/ibm/ILOG/CPLEX_Studio201/cpoptimizer"`

`export PATH="${PATH}:${CPLEX_HOME}/bin/x86-64_linux:${CPO_HOME}/bin/x86-64_linux"`

`export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CPLEX_HOME}/bin/x86-64_linux:${CPO_HOME}/bin/x86-64_linux"`

`export PYTHONPATH="${PYTHONPATH}:/opt/ibm/ILOG/CPLEX_Studio201/cplex/python/3.8/x86-64_linux"`
