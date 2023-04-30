#!/usr/bin/env python3
import sys
from tapestry_solver import get_expression
import minisat


def default_usage():
    # The argument must reference an instance file and the second
    print("Usage:", sys.argv[0], "INSTANCE_FILE", file=sys.stderr)
    exit(1)


def get_row(grid, i):
    return grid[i]


def get_column(grid, j):
    out = []
    for i in range(len(grid)):
        out.append(grid[i][j])
    return out


def read_instance(instance_file):
    file = open(instance_file)
    size = int(file.readline().split(' ')[0])
    fixed = []
    line = file.readline()
    while line.strip():
        fixed.append((int(line.split(' ')[0]), int(line.split(' ')[1]), int(line.split(' ')[2]), int(line.split(' ')[3])))
        line = file.readline()

    print(fixed)
    return size, fixed


def get_val_from_index(index, size):
    index -= 1
    quotient, color_ind = divmod(abs(index), size)
    quotient, shape_ind = divmod(quotient, size)
    row_ind, column_ind = divmod(quotient, size)
    return row_ind, column_ind, shape_ind, color_ind


def verify_solution(grid, size, fixed_cells):
    clean = True
    shape_square = [[elem[0] for elem in row] for row in grid]
    color_square = [[elem[1] for elem in row] for row in grid]

    for cell in fixed_cells:
        if shape_square[cell[0]][cell[1]] != cell[2]:
            clean = False
            print(f"FAIL. Shape {cell[2]} is required in Grid[{cell[0]}][{cell[1]}]")
        if color_square[cell[0]][cell[1]] != cell[3]:
            clean = False
            print(f"FAIL. Color {cell[3]} is required in Grid[{cell[0]}][{cell[1]}]")

    for i in range(size):
        for val in range(size):
            if val not in get_row(shape_square, i):
                clean = False
                print(f"FAIL. Shape {val} is missing in row {i}")
            if val not in get_column(shape_square, i):
                clean = False
                print(f"FAIL. Shape {val} is missing in column {i}")
            if val not in get_row(color_square, i):
                clean = False
                print(f"FAIL. Color {val} is missing in row {i}")
            if val not in get_column(color_square, i):
                clean = False
                print(f"FAIL. Color {val} is missing in column {i}")

        for j in range(size):
            for k in range(i, size) :
                for l in range(j, size) :
                    if (i != k or j != l) and grid[i][j] == grid[k][l]:
                        clean = False
                        print(f"FAIL. Pair {grid[i][j]} is not unique")

    if clean:
        print("SOLVED")
    else:
        print("FAIL")

    return clean


if __name__ == "__main__":
    if len(sys.argv) != 2:
        default_usage()

    size, fixed_cells = read_instance(sys.argv[1])
    n_rows = n_columns = n_shapes = n_colors = size
    expression = get_expression(size, fixed_cells)
    nb_vars = n_rows * n_columns * n_shapes * n_colors
    solution = minisat.minisat(nb_vars, [clause.minisat_str() for clause in expression], './minisatMac')

    if solution is None:
        print("The problem is unfeasible")
        exit(0)
    grid = [[(0, 0) for _ in range(size)] for _ in range(size)]
    list_sol = []
    for s in solution:
        row, column, shape, color = get_val_from_index(s, size)
        grid[row][column] = (shape, color)
        list_sol.append((row, column, shape, color))

    verify_solution(grid, size, fixed_cells)
        
    for row in grid:
        print(row)

