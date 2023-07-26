import pygame
from utils import get_triangle_points, get_cell_center
from typing import Tuple
from drl_lib.do_not_touch.contracts import MDPEnv
from envs.grid_world_mdp import GridWorldEnv
from envs.line_world_mdp import LineWorldEnv
from algorithms.dynamic_programming.policy_iteration import policy_iteration
from algorithms.dynamic_programming.value_iteration import value_iteration
from drl_lib.do_not_touch.result_structures import PolicyAndValueFunction


def launch_visualize_mdp(environment: MDPEnv,
                         HEIGHT: int, WIDTH: int,
                         win: Tuple[int, int], loose: Tuple[int, int],
                         pavf: PolicyAndValueFunction):

    assert (HEIGHT >= 1 and WIDTH > 2)
    pygame.init()

    # Define control of the world
    if len(environment.actions()) > 2:
        control_top = 0
        control_right = 1
        control_bottom = 2
        control_left = 3
    else:
        control_right = 1
        control_left = 0
        control_top = None
        control_bottom = None

    # Define some colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (169, 169, 169)

    # This sets the margin between each cell
    MARGIN = 3

    # Create a 2D array. A two dimensional array is simply a list of lists.
    grid = []
    for row in range(HEIGHT):
        grid.append([])
        for column in range(WIDTH):
            if (row, column) == win:
                grid[row].append(1)
            elif (row, column) == loose:
                grid[row].append(-1)
            else:
                grid[row].append(0)

    font = pygame.font.SysFont("Arial", 12)
    font_value = pygame.font.SysFont("Arial", 8)

    # Set the HEIGHT and WIDTH of the screen
    CELL_SIZE = {"width": 128, "height": 128}
    WINDOW_SIZE = [CELL_SIZE["width"] * WIDTH, CELL_SIZE["height"] * HEIGHT]
    screen = pygame.display.set_mode(WINDOW_SIZE)
    done = False

    # --- Main event loop
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # User clicked close
                done = True

        # Draw the grid
        for row in range(HEIGHT):
            for column in range(WIDTH):
                # Cells sizing
                cell_left = (CELL_SIZE['width'] * column)
                cell_top = (CELL_SIZE['height'] * row)
                cell_width = CELL_SIZE['width']
                cell_height = CELL_SIZE['height']

                cell_middle = get_cell_center(row, column, cell_width, cell_height)

                # Color cells when its win cell or loose cell
                if grid[row][column] == 1:
                    color = GREEN
                elif grid[row][column] == -1:
                    color = RED
                else:
                    color = WHITE

                current_rect = pygame.Rect(cell_left, cell_top, cell_width, cell_height)

                pygame.draw.rect(screen, color, current_rect)

                # State display
                state = row * WIDTH + column

                state_txt = font.render(f's = {str(state)}', True, BLACK)

                screen.blit(state_txt, (cell_middle[0] - state_txt.get_width() // 2,
                                        cell_middle[1] - state_txt.get_height() // 2))

                # State display

                value_txt = font_value.render(f'{round(pavf.v[state], 2)}', True, BLUE)

                screen.blit(value_txt, (cell_middle[0] - value_txt.get_width() // 2,
                                        (cell_middle[1] + 30) - value_txt.get_height() // 2))

                # Direction triangles
                top_t, right_t, bottom_t, left_t = get_triangle_points(row, column, cell_width, cell_height)

                color_top = GRAY
                color_right = GRAY
                color_bottom = GRAY
                color_left = GRAY

                if pavf.pi[state] == control_top:  # Top
                    color_top = BLUE
                elif pavf.pi[state] == control_right:  # Right
                    color_right = BLUE
                elif pavf.pi[state] == control_bottom:  # Bottom
                    color_bottom = BLUE
                elif pavf.pi[state] == control_left:  # Left
                    color_left = BLUE

                if len(environment.actions()) > 2:
                    pygame.draw.polygon(screen, color_top, top_t)
                    pygame.draw.polygon(screen, color_bottom, bottom_t)

                pygame.draw.polygon(screen, color_right, right_t)
                pygame.draw.polygon(screen, color_left, left_t)

        # Grid axes
        for row in range(1, HEIGHT):
            pygame.draw.line(screen, BLACK, (0, row * CELL_SIZE["height"] + 1),
                             (WINDOW_SIZE[0], row * CELL_SIZE["height"] + 1), MARGIN)

        for column in range(1, WIDTH):
            pygame.draw.line(screen, BLACK, (column * CELL_SIZE["width"] + 1, 0),
                             (column * CELL_SIZE["width"] + 1, WINDOW_SIZE[1]), MARGIN)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    grid_world = GridWorldEnv(5, 5, (4, 4), (0, 4))
    pavf = value_iteration(grid_world, gamma=0.99, theta=0.01)
    launch_visualize_mdp(grid_world,
                         grid_world.rows,
                         grid_world.cols,
                         grid_world.win_coords,
                         grid_world.loose_coords,
                         pavf)
