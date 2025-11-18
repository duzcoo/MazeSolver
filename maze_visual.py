import pygame
import random
import time
from collections import deque
import heapq
from typing import List, Tuple, Dict

# Type aliases for cleaner code
Maze = List[List[str]]
Coord = Tuple[int, int]

def create_random_maze(rows: int, cols: int, difficulty: str = "easy") -> Tuple[Maze, Coord, Coord]:
    """Generates a random maze with guaranteed path from start to goal.
    
    Difficulty affects how challenging the maze is:
    - Easy: Smaller grid, short direct path with minimal branching
    - Medium: Larger grid, moderate path length with some dead ends
    - Hard: Largest grid, long winding path with many dead ends and branches
    
    Harder difficulties have MORE open space with MORE obstacles to explore,
    not more walls blocking the path.
    """
    # Difficulty settings: controls path complexity and branching density
    difficulty_settings = {
        "easy": {
            "path_extension": 0.2,      # Minimal path extension
            "dead_end_prob": 0.08,      # Few dead ends
            "branch_prob": 0.12,        # Minimal branching
            "additional_paths": 0.0     # No extra paths
        },
        "medium": {
            "path_extension": 0.5,      # Moderate path extension
            "dead_end_prob": 0.20,      # Some dead ends
            "branch_prob": 0.25,        # Moderate branching
            "additional_paths": 0.15    # Some extra connecting paths
        },
        "hard": {
            "path_extension": 0.8,      # Significant path extension
            "dead_end_prob": 0.35,      # Many dead ends
            "branch_prob": 0.40,        # Heavy branching
            "additional_paths": 0.30    # More extra paths creating complexity
        }
    }
    
    settings = difficulty_settings.get(difficulty, difficulty_settings["easy"])
    
    # Start with all walls
    maze_grid = [['#' for _ in range(cols)] for _ in range(rows)]
    
    # Always place start and goal in corners (with padding)
    start_pos = (1, 1)
    goal_pos = (rows - 2, cols - 2)
    
    # Create a guaranteed path from start to goal first
    path = _generate_guaranteed_path(start_pos, goal_pos, rows, cols, settings["path_extension"])
    
    # Mark path cells as empty
    for row, col in path:
        maze_grid[row][col] = '.'
    
    # Add branching paths and dead ends based on difficulty
    _add_branches_and_dead_ends(maze_grid, path, settings["dead_end_prob"], 
                                settings["branch_prob"], rows, cols)
    
    # Add additional connecting paths for harder difficulties (creates more exploration options)
    if settings["additional_paths"] > 0:
        _add_connecting_paths(maze_grid, path, settings["additional_paths"], rows, cols)
    
    # Place start and goal markers
    maze_grid[start_pos[0]][start_pos[1]] = 'S'
    maze_grid[goal_pos[0]][goal_pos[1]] = 'G'
    
    return maze_grid, start_pos, goal_pos

def _generate_guaranteed_path(start: Coord, goal: Coord, rows: int, cols: int, extension_factor: float) -> List[Coord]:
    """Creates a guaranteed path from start to goal, with optional extensions for difficulty."""
    path = [start]
    current = start
    visited = {start}
    
    target_row, target_col = goal
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Calculate minimum steps needed and max steps allowed for extension
    min_distance = abs(target_row - start[0]) + abs(target_col - start[1])
    max_steps = int(min_distance * (1 + extension_factor * 2.5))
    steps = 0
    
    while current != goal and steps < max_steps:
        steps += 1
        current_row, current_col = current
        
        # Direction towards goal
        row_diff = target_row - current_row
        col_diff = target_col - current_col
        
        # Build candidate moves - prioritize moving towards goal
        candidates = []
        
        # Primary moves: directly towards goal
        if row_diff != 0:
            candidates.append((current_row + (1 if row_diff > 0 else -1), current_col, 1))
        if col_diff != 0:
            candidates.append((current_row, current_col + (1 if col_diff > 0 else -1), 1))
        
        # Secondary moves: perpendicular (for path extension)
        if extension_factor > 0.3:
            if row_diff == 0:  # Moving horizontally, allow vertical detour
                if 1 < current_row < rows - 2:
                    candidates.extend([
                        (current_row + 1, current_col, 2),
                        (current_row - 1, current_col, 2)
                    ])
            if col_diff == 0:  # Moving vertically, allow horizontal detour
                if 1 < current_col < cols - 2:
                    candidates.extend([
                        (current_row, current_col + 1, 2),
                        (current_row, current_col - 1, 2)
                    ])
            
            # Add all perpendicular moves for more wandering
            if abs(row_diff) > 2:  # Far vertically, allow horizontal moves
                candidates.extend([
                    (current_row, current_col + 1, 3),
                    (current_row, current_col - 1, 3)
                ])
            if abs(col_diff) > 2:  # Far horizontally, allow vertical moves
                candidates.extend([
                    (current_row + 1, current_col, 3),
                    (current_row - 1, current_col, 3)
                ])
        
        # Filter valid candidates (in bounds, not visited)
        valid_candidates = [
            (r, c) for r, c, priority in candidates
            if 1 <= r < rows - 1 and 1 <= c < cols - 1 and (r, c) not in visited
        ]
        
        # If no valid candidates, try any adjacent cell
        if not valid_candidates:
            for dr, dc in directions:
                next_row = current_row + dr
                next_col = current_col + dc
                if (1 <= next_row < rows - 1 and 1 <= next_col < cols - 1 and
                    (next_row, next_col) not in visited):
                    valid_candidates.append((next_row, next_col))
        
        if valid_candidates:
            # Sort by distance to goal
            valid_candidates.sort(key=lambda pos: abs(pos[0] - target_row) + abs(pos[1] - target_col))
            
            # Choose move based on difficulty and extension factor
            if len(valid_candidates) > 1 and random.random() < extension_factor:
                # For harder difficulties, sometimes take a less optimal path
                # This creates longer, more winding paths
                choice_index = random.randint(0, min(2, len(valid_candidates) - 1))
                current = valid_candidates[choice_index]
            else:
                # Usually take the best option
                current = valid_candidates[0]
            
            path.append(current)
            visited.add(current)
        else:
            # Shouldn't happen, but if stuck, backtrack or force move to goal
            if row_diff != 0:
                current = (current_row + (1 if row_diff > 0 else -1), current_col)
            elif col_diff != 0:
                current = (current_row, current_col + (1 if col_diff > 0 else -1))
            
            if current not in visited:
                path.append(current)
                visited.add(current)
    
    # Final check: ensure we reach goal
    if path[-1] != goal:
        # Connect directly if needed
        last = path[-1]
        last_row, last_col = last
        
        # Create direct connection to goal
        while last_row != target_row or last_col != target_col:
            if last_row < target_row:
                last_row += 1
            elif last_row > target_row:
                last_row -= 1
            elif last_col < target_col:
                last_col += 1
            elif last_col > target_col:
                last_col -= 1
            
            new_cell = (last_row, last_col)
            if new_cell not in visited:
                path.append(new_cell)
                visited.add(new_cell)
    
    return path

def _add_branches_and_dead_ends(maze_grid: Maze, main_path: List[Coord], dead_end_prob: float, 
                                branch_prob: float, rows: int, cols: int):
    """Adds branches and dead ends to make the maze more challenging.
    
    These create obstacles to explore - they waste algorithm time exploring
    dead ends, making the maze harder without blocking the solution path.
    """
    path_set = set(main_path)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    open_cells = set(main_path)  # Track all open cells to avoid overlapping branches
    
    # For each cell in the main path, potentially add a branch
    for path_cell in main_path:
        if random.random() < branch_prob:
            # Try to create a branch from this point
            row, col = path_cell
            shuffled_dirs = directions.copy()
            random.shuffle(shuffled_dirs)
            
            for dr, dc in shuffled_dirs:
                branch_start = (row + dr, col + dc)
                branch_row, branch_col = branch_start
                
                # Check if valid starting point for branch (wall, not in path, not already open)
                if (1 <= branch_row < rows - 1 and 1 <= branch_col < cols - 1 and
                    branch_start not in path_set and branch_start not in open_cells and
                    maze_grid[branch_row][branch_col] == '#'):
                    
                    # Create a dead end branch
                    branch_length = random.randint(2, int(5 + dead_end_prob * 12))
                    current = branch_start
                    branch_cells = []
                    
                    for _ in range(branch_length):
                        branch_cells.append(current)
                        open_cells.add(current)
                        maze_grid[current[0]][current[1]] = '.'
                        
                        # Try to continue in same direction or turn
                        next_options = []
                        for ndr, ndc in directions:
                            next_cell = (current[0] + ndr, current[1] + ndc)
                            next_row, next_col = next_cell
                            
                            if (1 <= next_row < rows - 1 and 1 <= next_col < cols - 1 and
                                next_cell not in path_set and next_cell not in branch_cells and
                                maze_grid[next_row][next_col] == '#'):
                                next_options.append(next_cell)
                        
                        if not next_options:
                            break
                        
                        # Prefer continuing in similar direction
                        if random.random() < 0.6 and len(next_options) > 1:
                            # Continue roughly in same direction
                            preferred = [(current[0] + dr, current[1] + dc) for dr, dc in directions
                                        if (current[0] + dr, current[1] + dc) in next_options]
                            if preferred:
                                current = random.choice(preferred)
                            else:
                                current = random.choice(next_options)
                        else:
                            current = random.choice(next_options)
                    
                    break  # Only create one branch per path cell

def _add_connecting_paths(maze_grid: Maze, main_path: List[Coord], path_prob: float, rows: int, cols: int):
    """Adds additional connecting paths between parts of the main path.
    
    These create alternative routes that increase exploration complexity
    without blocking the solution - algorithms need to explore more options.
    """
    path_set = set(main_path)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Try to connect different segments of the main path
    path_segments = []
    segment_size = max(3, len(main_path) // 10)
    
    for i in range(0, len(main_path) - segment_size, segment_size):
        segment = main_path[i:i + segment_size]
        if len(segment) >= 2:
            path_segments.append((segment[0], segment[-1]))
    
    # Try to create connecting paths between segments
    for start_seg, end_seg in path_segments:
        if random.random() < path_prob and len(path_segments) > 1:
            # Pick a random point on start segment
            start_point = random.choice([start_seg, end_seg])
            
            # Find a point further along the path
            if start_point == start_seg:
                target_idx = min(len(main_path) - 1, main_path.index(start_seg) + segment_size * 2)
            else:
                target_idx = max(0, main_path.index(end_seg) - segment_size * 2)
            
            if 0 <= target_idx < len(main_path):
                target_point = main_path[target_idx]
                
                # Try to create a path between these points
                current = start_point
                visited = {start_point}
                path_length = 0
                max_path_length = abs(target_point[0] - start_point[0]) + abs(target_point[1] - start_point[1]) + 3
                
                while current != target_point and path_length < max_path_length:
                    path_length += 1
                    current_row, current_col = current
                    
                    # Move towards target
                    row_diff = target_point[0] - current_row
                    col_diff = target_point[1] - current_col
                    
                    candidates = []
                    if row_diff != 0:
                        candidates.append((current_row + (1 if row_diff > 0 else -1), current_col))
                    if col_diff != 0:
                        candidates.append((current_row, current_col + (1 if col_diff > 0 else -1)))
                    
                    # Add perpendicular options for variety
                    if abs(row_diff) > 1:
                        candidates.extend([(current_row, current_col + 1), (current_row, current_col - 1)])
                    if abs(col_diff) > 1:
                        candidates.extend([(current_row + 1, current_col), (current_row - 1, current_col)])
                    
                    # Filter valid candidates
                    valid = [
                        (r, c) for r, c in candidates
                        if 1 <= r < rows - 1 and 1 <= c < cols - 1 and (r, c) not in visited
                    ]
                    
                    if not valid:
                        break
                    
                    # Sort by distance to target
                    valid.sort(key=lambda pos: abs(pos[0] - target_point[0]) + abs(pos[1] - target_point[1]))
                    
                    # Sometimes take a less direct route
                    if len(valid) > 1 and random.random() < 0.3:
                        current = random.choice(valid[:len(valid)//2 + 1])
                    else:
                        current = valid[0]
                    
                    visited.add(current)
                    if maze_grid[current[0]][current[1]] == '#':
                        maze_grid[current[0]][current[1]] = '.'

def find_neighbors(maze: Maze, row: int, col: int) -> List[Coord]:
    """Gets all adjacent walkable cells (no diagonals)."""
    neighbors_list = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    for delta_row, delta_col in directions:
        next_row = row + delta_row
        next_col = col + delta_col
        
        # Make sure we're in bounds and not hitting a wall
        if (0 <= next_row < len(maze) and 
            0 <= next_col < len(maze[0]) and 
            maze[next_row][next_col] != '#'):
            neighbors_list.append((next_row, next_col))
    
    return neighbors_list

def build_path_from_parents(parent_map: Dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
    """Walks backwards through parent pointers to reconstruct the full path."""
    # Can't build a path if the goal was never reached
    if goal not in parent_map and goal != start:
        return []
    
    # Trace backwards from goal to start
    current = goal
    path_backwards = [current]
    
    while current != start:
        current = parent_map[current]
        path_backwards.append(current)
    
    # Flip it around since we built it in reverse
    path_backwards.reverse()
    return path_backwards

def solve_bfs(maze: Maze, start: Coord, goal: Coord):
    """BFS solver - returns path, exploration order, and performance stats."""
    start_time = time.perf_counter()
    
    queue = deque([start])
    parent_map: Dict[Coord, Coord] = {}
    visited_nodes = {start}
    exploration_sequence: List[Coord] = []  # Track order for visualization
    
    while queue:
        current = queue.popleft()
        exploration_sequence.append(current)
        
        # Success!
        if current == goal:
            break
        
        # Check all neighboring cells
        for neighbor in find_neighbors(maze, *current):
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                parent_map[neighbor] = current
                queue.append(neighbor)
    
    solution_path = build_path_from_parents(parent_map, start, goal)
    
    # Calculate stats
    elapsed_time = (time.perf_counter() - start_time) * 1000.0  # Convert to milliseconds
    path_length = len(solution_path) if solution_path else 0
    nodes_explored = len(exploration_sequence)
    
    stats = {
        'path_length': path_length,
        'nodes_explored': nodes_explored,
        'runtime_ms': elapsed_time
    }
    
    return solution_path, exploration_sequence, stats

def manhattan_heuristic(point1: Coord, point2: Coord) -> int:
    """Simple Manhattan distance - works well for grid-based mazes."""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def solve_astar(maze: Maze, start: Coord, goal: Coord):
    """A* solver with Manhattan heuristic - returns path, exploration order, and performance stats."""
    start_time = time.perf_counter()
    
    # Priority queue: (f_score, tie_breaker, position)
    frontier = []
    initial_heuristic = manhattan_heuristic(start, goal)
    heapq.heappush(frontier, (initial_heuristic, 0, start))
    
    parent_map: Dict[Coord, Coord] = {}
    g_costs: Dict[Coord, int] = {start: 0}  # Actual cost from start
    nodes_in_frontier = {start}
    exploration_sequence: List[Coord] = []
    tie_breaker_counter = 0  # Prevents heap comparison issues
    
    while frontier:
        f_score, _, current = heapq.heappop(frontier)
        exploration_sequence.append(current)
        
        if current == goal:
            break
        
        # Evaluate each neighbor
        for neighbor in find_neighbors(maze, *current):
            cost_through_current = g_costs[current] + 1
            
            # Only update if we found a better path
            if cost_through_current < g_costs.get(neighbor, float('inf')):
                parent_map[neighbor] = current
                g_costs[neighbor] = cost_through_current
                h_score = manhattan_heuristic(neighbor, goal)
                new_f_score = cost_through_current + h_score
                
                # Add to frontier if it's new
                if neighbor not in nodes_in_frontier:
                    tie_breaker_counter += 1
                    heapq.heappush(frontier, (new_f_score, tie_breaker_counter, neighbor))
                    nodes_in_frontier.add(neighbor)
    
    solution_path = build_path_from_parents(parent_map, start, goal)
    
    # Calculate stats
    elapsed_time = (time.perf_counter() - start_time) * 1000.0  # Convert to milliseconds
    path_length = len(solution_path) if solution_path else 0
    nodes_explored = len(exploration_sequence)
    
    stats = {
        'path_length': path_length,
        'nodes_explored': nodes_explored,
        'runtime_ms': elapsed_time
    }
    
    return solution_path, exploration_sequence, stats

# Visualization constants - tweak these if you want different sizes
BASE_CELL_SIZE = 24
CELL_SPACING = 2
MAX_WINDOW_WIDTH = 1920  # Maximum window width to keep UI manageable

# Base maze size - will scale up for harder difficulties
BASE_MAZE_ROWS = 20
BASE_MAZE_COLS = 30

def get_maze_size(difficulty: str) -> Tuple[int, int]:
    """Returns maze dimensions based on difficulty level."""
    size_multipliers = {
        "easy": (1.0, 1.0),      # 20x30
        "medium": (1.4, 1.4),    # 28x42
        "hard": (1.8, 1.8)       # 36x54
    }
    multiplier = size_multipliers.get(difficulty, (1.0, 1.0))
    rows = int(BASE_MAZE_ROWS * multiplier[0])
    cols = int(BASE_MAZE_COLS * multiplier[1])
    return rows, cols

def get_cell_size(rows: int, cols: int, comparison_mode: bool = False) -> int:
    """Calculates appropriate cell size to keep window manageable.
    
    In comparison mode (top/bottom layout), scales down cell size if needed 
    to prevent window from becoming too large.
    """
    height_multiplier = 2 if comparison_mode else 1
    desired_height = rows * BASE_CELL_SIZE * height_multiplier
    desired_width = cols * BASE_CELL_SIZE
    
    # Check both dimensions and scale to fit
    max_height = 1080  # Maximum window height
    
    if desired_height > max_height:
        # Scale down to fit within max height
        cell_size = int(max_height / (rows * height_multiplier))
    elif desired_width > MAX_WINDOW_WIDTH:
        # Scale down to fit within max width
        cell_size = int(MAX_WINDOW_WIDTH / cols)
    else:
        return BASE_CELL_SIZE
    
    # Don't make cells too small - minimum 12 pixels
    return max(12, cell_size)

# Color palette - feel free to customize these
COLOR_BACKGROUND = (0, 0, 0)
COLOR_EMPTY = (255, 255, 255)
COLOR_WALL = (0, 0, 0)
COLOR_EXPLORED = (0, 120, 255)      # Nice blue for explored nodes
COLOR_PATH = (255, 215, 0)          # Gold/yellow for final path
COLOR_START = (0, 200, 0)           # Green for start
COLOR_GOAL = (220, 0, 0)            # Red for goal
COLOR_TEXT = (160, 32, 240)         # Purple for UI text

def render_maze(screen, maze: Maze, explored: List[Coord], path: List[Coord], 
                start: Coord, goal: Coord, animation_step: int, x_offset: int = 0, 
                y_offset: int = 0, cell_size: int = BASE_CELL_SIZE):
    """Draws the maze with exploration animation and solution path.
    
    Args:
        screen: Pygame surface to draw on
        maze: The maze grid
        explored: List of explored coordinates in order
        path: The solution path
        start: Start position
        goal: Goal position
        animation_step: Current animation frame
        x_offset: X offset for positioning (default 0)
        y_offset: Y offset for top/bottom layout (default 0)
        cell_size: Size of each cell in pixels (default BASE_CELL_SIZE)
    """
    # Figure out which cells should be visible at this animation step
    explored_up_to_step = set(explored[:animation_step])
    
    # Only show the path after we've finished exploring
    should_show_path = animation_step >= len(explored)
    path_cells = set(path) if should_show_path else set()
    
    # Draw each cell
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            pixel_x = x_offset + col * cell_size
            pixel_y = y_offset + row * cell_size
            cell_type = maze[row][col]
            
            # Determine what color this cell should be
            cell_color = COLOR_EMPTY
            
            if cell_type == '#':
                cell_color = COLOR_WALL
            elif (row, col) == start:
                cell_color = COLOR_START
            elif (row, col) == goal:
                cell_color = COLOR_GOAL
            elif (row, col) in path_cells:
                # Path overrides explored color
                cell_color = COLOR_PATH
            elif (row, col) in explored_up_to_step:
                cell_color = COLOR_EXPLORED
            
            # Draw the cell rectangle
            pygame.draw.rect(screen, cell_color, 
                           (pixel_x, pixel_y, 
                            cell_size - CELL_SPACING, 
                            cell_size - CELL_SPACING))

def main():
    """Main game loop - handles input, animation, and rendering."""
    pygame.init()
    
    # Current settings - must be set before window size calculation
    current_algorithm = "astar"
    current_difficulty = "easy"
    
    # Window size depends on mode and difficulty - comparison mode needs double height (top/bottom)
    comparison_mode = False
    current_rows, current_cols = get_maze_size(current_difficulty)
    
    # Calculate cell size based on mode and maze size
    current_cell_size = get_cell_size(current_rows, current_cols, comparison_mode)
    
    base_width = current_cols * current_cell_size
    base_height = current_rows * current_cell_size
    window_width = base_width
    window_height = base_height * 2 if comparison_mode else base_height
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Maze Solver Visualization - BFS vs A*")
    
    clock = pygame.time.Clock()
    
    def create_new_maze_and_solve():
        """Helper to generate a maze and solve it with current algorithm or both."""
        maze_rows, maze_cols = get_maze_size(current_difficulty)
        new_maze, start_pos, goal_pos = create_random_maze(maze_rows, maze_cols, current_difficulty)
        
        if comparison_mode:
            # Solve with both algorithms simultaneously
            bfs_solution, bfs_explored, bfs_stats = solve_bfs(new_maze, start_pos, goal_pos)
            astar_solution, astar_explored, astar_stats = solve_astar(new_maze, start_pos, goal_pos)
            
            # Print stats for both
            print(f"\n=== Comparison Mode ===")
            print(f"\nBFS Results:")
            print(f"  Path length: {bfs_stats['path_length']} steps")
            print(f"  Nodes explored: {bfs_stats['nodes_explored']}")
            print(f"  Runtime: {bfs_stats['runtime_ms']:.2f} ms")
            print(f"  Status: {'Path found' if bfs_stats['path_length'] > 0 else 'No path found'}")
            
            print(f"\nA* Results:")
            print(f"  Path length: {astar_stats['path_length']} steps")
            print(f"  Nodes explored: {astar_stats['nodes_explored']}")
            print(f"  Runtime: {astar_stats['runtime_ms']:.2f} ms")
            print(f"  Status: {'Path found' if astar_stats['path_length'] > 0 else 'No path found'}")
            
            print(f"\nComparison:")
            if bfs_stats['nodes_explored'] > 0 and astar_stats['nodes_explored'] > 0:
                efficiency = ((bfs_stats['nodes_explored'] - astar_stats['nodes_explored']) / 
                             bfs_stats['nodes_explored']) * 100
                print(f"  A* explored {abs(efficiency):.1f}% {'fewer' if efficiency > 0 else 'more'} nodes")
            
            return (new_maze, start_pos, goal_pos, 
                   bfs_solution, bfs_explored, bfs_stats,
                   astar_solution, astar_explored, astar_stats)
        else:
            # Single algorithm mode
            if current_algorithm == "bfs":
                solution, exploration_order, stats = solve_bfs(new_maze, start_pos, goal_pos)
            else:
                solution, exploration_order, stats = solve_astar(new_maze, start_pos, goal_pos)
            
            # Print stats to console
            algo_name = current_algorithm.upper()
            print(f"\n{algo_name} Results:")
            print(f"  Path length: {stats['path_length']} steps")
            print(f"  Nodes explored: {stats['nodes_explored']}")
            print(f"  Runtime: {stats['runtime_ms']:.2f} ms")
            if stats['path_length'] == 0:
                print(f"  Status: No path found!")
            else:
                print(f"  Status: Path found successfully")
            
            return new_maze, start_pos, goal_pos, solution, exploration_order, stats
    
    # Initial setup
    if comparison_mode:
        maze, start, goal, bfs_path, bfs_explored, bfs_stats, astar_path, astar_explored, astar_stats = create_new_maze_and_solve()
        bfs_step = 0
        astar_step = 0
    else:
        maze, start, goal, path, explored, stats = create_new_maze_and_solve()
        animation_step = 0
    
    is_running = True
    
    # Font for on-screen info
    info_font = pygame.font.SysFont(None, 20)
    
    while is_running:
        clock.tick(30)  # Cap at 30 FPS for smooth animation
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    is_running = False
                elif event.key == pygame.K_c:
                    # Toggle comparison mode
                    comparison_mode = not comparison_mode
                    current_rows, current_cols = get_maze_size(current_difficulty)
                    current_cell_size = get_cell_size(current_rows, current_cols, comparison_mode)
                    base_width = current_cols * current_cell_size
                    base_height = current_rows * current_cell_size
                    window_width = base_width
                    window_height = base_height * 2 if comparison_mode else base_height
                    screen = pygame.display.set_mode((window_width, window_height))
                    
                    # Regenerate with new mode
                    if comparison_mode:
                        maze, start, goal, bfs_path, bfs_explored, bfs_stats, astar_path, astar_explored, astar_stats = create_new_maze_and_solve()
                        bfs_step = 0
                        astar_step = 0
                    else:
                        maze, start, goal, path, explored, stats = create_new_maze_and_solve()
                        animation_step = 0
                elif event.key == pygame.K_b and not comparison_mode:
                    # Switch to BFS (only in single mode)
                    current_algorithm = "bfs"
                    maze, start, goal, path, explored, stats = create_new_maze_and_solve()
                    animation_step = 0
                elif event.key == pygame.K_a and not comparison_mode:
                    # Switch to A* (only in single mode)
                    current_algorithm = "astar"
                    maze, start, goal, path, explored, stats = create_new_maze_and_solve()
                    animation_step = 0
                elif event.key == pygame.K_1:
                    # Easy difficulty
                    current_difficulty = "easy"
                    # Resize window for new difficulty
                    current_rows, current_cols = get_maze_size(current_difficulty)
                    current_cell_size = get_cell_size(current_rows, current_cols, comparison_mode)
                    base_width = current_cols * current_cell_size
                    base_height = current_rows * current_cell_size
                    window_width = base_width
                    window_height = base_height * 2 if comparison_mode else base_height
                    screen = pygame.display.set_mode((window_width, window_height))
                    
                    if comparison_mode:
                        maze, start, goal, bfs_path, bfs_explored, bfs_stats, astar_path, astar_explored, astar_stats = create_new_maze_and_solve()
                        bfs_step = 0
                        astar_step = 0
                    else:
                        maze, start, goal, path, explored, stats = create_new_maze_and_solve()
                        animation_step = 0
                elif event.key == pygame.K_2:
                    # Medium difficulty
                    current_difficulty = "medium"
                    # Resize window for new difficulty
                    current_rows, current_cols = get_maze_size(current_difficulty)
                    current_cell_size = get_cell_size(current_rows, current_cols, comparison_mode)
                    base_width = current_cols * current_cell_size
                    base_height = current_rows * current_cell_size
                    window_width = base_width
                    window_height = base_height * 2 if comparison_mode else base_height
                    screen = pygame.display.set_mode((window_width, window_height))
                    
                    if comparison_mode:
                        maze, start, goal, bfs_path, bfs_explored, bfs_stats, astar_path, astar_explored, astar_stats = create_new_maze_and_solve()
                        bfs_step = 0
                        astar_step = 0
                    else:
                        maze, start, goal, path, explored, stats = create_new_maze_and_solve()
                        animation_step = 0
                elif event.key == pygame.K_3:
                    # Hard difficulty
                    current_difficulty = "hard"
                    # Resize window for new difficulty
                    current_rows, current_cols = get_maze_size(current_difficulty)
                    current_cell_size = get_cell_size(current_rows, current_cols, comparison_mode)
                    base_width = current_cols * current_cell_size
                    base_height = current_rows * current_cell_size
                    window_width = base_width
                    window_height = base_height * 2 if comparison_mode else base_height
                    screen = pygame.display.set_mode((window_width, window_height))
                    
                    if comparison_mode:
                        maze, start, goal, bfs_path, bfs_explored, bfs_stats, astar_path, astar_explored, astar_stats = create_new_maze_and_solve()
                        bfs_step = 0
                        astar_step = 0
                    else:
                        maze, start, goal, path, explored, stats = create_new_maze_and_solve()
                        animation_step = 0
                elif event.key == pygame.K_r:
                    # Regenerate with same settings
                    if comparison_mode:
                        maze, start, goal, bfs_path, bfs_explored, bfs_stats, astar_path, astar_explored, astar_stats = create_new_maze_and_solve()
                        bfs_step = 0
                        astar_step = 0
                    else:
                        maze, start, goal, path, explored, stats = create_new_maze_and_solve()
                        animation_step = 0
        
        # Clear screen
        screen.fill(COLOR_BACKGROUND)
        
        if comparison_mode:
            # Comparison mode - show both algorithms top and bottom
            # Advance animations independently
            bfs_max_steps = len(bfs_explored) + len(bfs_path)
            astar_max_steps = len(astar_explored) + len(astar_path)
            
            if bfs_step < bfs_max_steps:
                bfs_step += 1
            if astar_step < astar_max_steps:
                astar_step += 1
            
            # Draw BFS on the top
            render_maze(screen, maze, bfs_explored, bfs_path, start, goal, bfs_step, 
                       x_offset=0, y_offset=0, cell_size=current_cell_size)
            
            # Draw A* on the bottom
            render_maze(screen, maze, astar_explored, astar_path, start, goal, astar_step, 
                       x_offset=0, y_offset=base_height, cell_size=current_cell_size)
            
            # Draw divider line between top and bottom
            divider_y = base_height
            pygame.draw.line(screen, COLOR_TEXT, (0, divider_y), (window_width, divider_y), 2)
            
            # Display labels
            bfs_label = info_font.render("BFS", True, COLOR_TEXT)
            screen.blit(bfs_label, (10, 5))
            
            astar_label = info_font.render("A*", True, COLOR_TEXT)
            screen.blit(astar_label, (10, base_height + 5))
            
            # Only show stats after both animations complete
            both_complete = (bfs_step >= bfs_max_steps and astar_step >= astar_max_steps)
            
            if both_complete:
                # BFS stats in top right corner of BFS maze (top section)
                bfs_stats_lines = [
                    f"Path: {bfs_stats['path_length']} steps",
                    f"Explored: {bfs_stats['nodes_explored']}",
                    f"Time: {bfs_stats['runtime_ms']:.2f} ms"
                ]
                
                stats_start_y = 5
                padding = 10
                
                # Render all BFS stats to calculate max width for right alignment
                bfs_stat_surfaces = []
                max_bfs_width = 0
                for stat_line in bfs_stats_lines:
                    stat_surface = info_font.render(stat_line, True, COLOR_TEXT)
                    bfs_stat_surfaces.append(stat_surface)
                    max_bfs_width = max(max_bfs_width, stat_surface.get_width())
                
                # Position BFS stats from right edge
                bfs_stats_x = window_width - max_bfs_width - padding
                for i, stat_surface in enumerate(bfs_stat_surfaces):
                    screen.blit(stat_surface, (bfs_stats_x, stats_start_y + (i * 18)))
                
                # A* stats in top right corner of A* maze (bottom section)
                astar_stats_lines = [
                    f"Path: {astar_stats['path_length']} steps",
                    f"Explored: {astar_stats['nodes_explored']}",
                    f"Time: {astar_stats['runtime_ms']:.2f} ms"
                ]
                
                astar_stats_start_y = base_height + 5
                
                # Render all A* stats to calculate max width for right alignment
                astar_stat_surfaces = []
                max_astar_width = 0
                for stat_line in astar_stats_lines:
                    stat_surface = info_font.render(stat_line, True, COLOR_TEXT)
                    astar_stat_surfaces.append(stat_surface)
                    max_astar_width = max(max_astar_width, stat_surface.get_width())
                
                # Position A* stats from right edge
                astar_stats_x = window_width - max_astar_width - padding
                for i, stat_surface in enumerate(astar_stat_surfaces):
                    screen.blit(stat_surface, (astar_stats_x, astar_stats_start_y + (i * 18)))
            
            # Controls at bottom
            controls_text = f"C=comparison mode | 1/2/3=difficulty | R=regenerate | ESC=quit"
            controls_surface = info_font.render(controls_text, True, COLOR_TEXT)
            screen.blit(controls_surface, (5, window_height - 20))
        else:
            # Single algorithm mode
            # Advance animation if not done
            max_steps = len(explored) + len(path)
            if animation_step < max_steps:
                animation_step += 1
            
            # Draw everything
            render_maze(screen, maze, explored, path, start, goal, animation_step, cell_size=current_cell_size)
            
            # Display controls and current settings
            controls_text = (f"Algorithm: {current_algorithm.upper()} | "
                            f"Difficulty: {current_difficulty} | "
                            f"C=compare | B=BFS, A=A*, 1/2/3=difficulty, R=regenerate")
            text_surface = info_font.render(controls_text, True, COLOR_TEXT)
            screen.blit(text_surface, (5, 5))
            
            # Display performance stats on screen
            stats_y_offset = 25
            stats_lines = [
                f"Path length: {stats['path_length']} steps",
                f"Nodes explored: {stats['nodes_explored']}",
                f"Runtime: {stats['runtime_ms']:.2f} ms"
            ]
            
            for i, stat_line in enumerate(stats_lines):
                stat_surface = info_font.render(stat_line, True, COLOR_TEXT)
                screen.blit(stat_surface, (5, 5 + stats_y_offset + (i * 20)))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
