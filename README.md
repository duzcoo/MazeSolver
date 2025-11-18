# MazeSolver
Maze visualizer and solver that compares BFS and A* side by side. Generates a random maze, animates exploration, and reports path length, explored nodes, and runtime so you can see how heuristics change search efficiency.

## Prerequisites
- Python 3.9+ (tested with 3.11)
- `pip`

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pygame
```

## Run
```bash
python maze_visual.py
```

## Controls
- `C` toggle comparison mode (BFS left, A* right)
- `B` switch to BFS (single mode)
- `A` switch to A* (single mode)
- `1/2/3` set difficulty easy/medium/hard and regenerate
- `R` regenerate maze with current settings
- `ESC` quit

## Notes
- Start is green, goal is red, explored cells are blue, final path is gold.
- High difficulty can block the start/goal; if no path exists, stats will show “No path found.”
