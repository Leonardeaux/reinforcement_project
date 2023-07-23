def get_cell_center(row: int, col: int, cell_width: int, cell_height: int):

    x = col * cell_width + cell_width / 2
    y = row * cell_height + cell_height / 2

    return x, y


def get_triangle_points(row: int, col: int, cell_width: int, cell_height: int):

    center_x, center_y = get_cell_center(row, col, cell_width, cell_height)

    top_triangle_points = [(center_x, center_y - cell_height / 2), (center_x - cell_width / 4, center_y - 35),
                           (center_x + cell_width / 4, center_y - 35)]
    right_triangle_points = [(center_x + cell_width / 2, center_y), (center_x + 35, center_y - cell_height / 4),
                             (center_x + 35, center_y + cell_height / 4)]
    bottom_triangle_points = [(center_x, center_y + cell_height / 2), (center_x - cell_width / 4, center_y + 35),
                              (center_x + cell_width / 4, center_y + 35)]
    left_triangle_points = [(center_x - cell_width / 2, center_y), (center_x - 35, center_y - cell_height / 4),
                            (center_x - 35, center_y + cell_height / 4)]

    return top_triangle_points, right_triangle_points, bottom_triangle_points, left_triangle_points
