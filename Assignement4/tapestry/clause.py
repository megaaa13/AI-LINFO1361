"""
Class used to represent a clause in CNF for the tapestry problem.
Variable X_i_j_k_l is true iff cell at position (i,j) has shape k and color l on it.

For example, to create a clause:

X_0_1_0_0 or ~X_1_2_4_6 or X_3_3_1_2

you can do:

clause = Clause(7)
clause.add_positive(0, 1, 0, 0)
clause.add_negative(1, 2, 4, 6)
clause.add_positive(3, 3, 1, 2)

"""


class Clause:

    def __init__(self, size, varname='C'):
        self.varname = varname
        self.n_rows = self.n_columns = self.n_shapes = self.n_colors = size
        self.value = []

    def index(self, row_ind, column_ind, shape_ind, color_ind):
        if 0 <= row_ind < self.n_rows and 0 <= column_ind < self.n_columns and 0 <= shape_ind < self.n_shapes and 0 <= color_ind < self.n_colors:
            return row_ind * self.n_columns * self.n_shapes * self.n_colors + column_ind * self.n_shapes * self.n_colors + shape_ind * self.n_colors + color_ind
        else:
            raise ValueError("Indices : row_ind =", row_ind, "column_ind =", column_ind, "shape_ind =", shape_ind, "color_ind =", color_ind, "is incorrect")

    def str_from_index(self, index):
        if index >= 0:
            index -= 1
        else:
            index += 1
        quotient, color_ind = divmod(abs(index), self.n_colors)
        quotient, shape_ind = divmod(quotient, self.n_shapes)
        row_ind, column_ind = divmod(quotient, self.n_columns)

        if index < 0:
            return '~{0}_{1}_{2}_{3}_{4}'.format(self.varname, row_ind, column_ind, shape_ind, color_ind)
        return '{0}_{1}_{2}_{3}_{4}'.format(self.varname, row_ind, column_ind, shape_ind, color_ind)

    def add_positive(self, row_ind, column_ind, shape_ind, color_ind):
        self.value.append(self.index(row_ind, column_ind, shape_ind, color_ind)+1)

    def add_negative(self, row_ind, column_ind, shape_ind, color_ind):
        self.value.append(-self.index(row_ind, column_ind, shape_ind, color_ind)-1)

    def minisat_str(self):
        return ' '.join([str(x) for x in self.value])

    def __str__(self):
        return ' or '.join([self.str_from_index(x) for x in self.value])


if __name__ == '__main__':
    clause = Clause(5)
    clause.add_positive(1, 1, 1, 1)
    clause.add_negative(1, 2, 3, 4)
    clause.add_positive(2, 2, 2, 2)
    print(clause)
    print(clause.minisat_str())
