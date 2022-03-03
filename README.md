# RoadGenerator

![gif](https://media.giphy.com/media/HPsaIfGklJFDy2xysR/giphy.gif)

RoadGenerator is a tool for better test case generation in the Carolo-Cup setting. Given a fixed 2D space, it can

* generate all valid roads.
* generate roads matching given specifications (number of intersections, 90-degree turns, etc.).
* visualize and export the end result for use in simulation.

## Process Chain Overview

![process-chain](https://user-images.githubusercontent.com/17745868/156574755-74712b7e-15d2-4001-9f2d-95b43c2cb617.svg)

## Setup

`sudo apt install libpango1.0-dev pkg-config python3-dev`

`sudo apt-get install python3.8-dev` (change version)

`sudo apt install texlive-full`

`sudo apt install libcairo2-dev`

`sudo apt install ffmpeg`

`sudo apt-get install build-essential libgl1-mesa-dev`

`pip install -r requirements.txt`

### GUROBI

GUROBI is the preferred solver for this project.

Install license with tools from [here](https://support.gurobi.com/hc/en-us/articles/360059842732)

License key can be found [here]([https://www.gurobi.com/downloads/free-academic-license/)

Then install gurobi for python via
`python -m pip install gurobipy`

## Running

In Terminal (from project root):

`python -m manim -ps creation_scenes.py FMTrackSuperConfig`

Options:

* -p : __play__
* -ps : __generate last image of scene__
* -ql : __fast low quality render__
