from clause import *

"""
For the tapestry problem, the only code you have to do is in this file.

You should replace

# your code here

by a code generating a list of clauses modeling the queen problem
for the input file.

You should build clauses using the Clause class defined in clause.py

Read the comment on top of clause.py to see how this works.
"""


def get_expression(size, fixed_cells=None):

    expression = []

    # A Clause is a set of literals linked by an OR
    # All the clauses are linked by an AND


    for i in range(size):
        for j in range(size):
            for a in range(size):
                for b in range(size):
                    # Here we have to add the all the possbile values for each cell
                    pass


    # Add the fixed cells
    if fixed_cells is not None:
        for i in range(len(fixed_cells)):
            clause = Clause(size)
            clause.add_positive(fixed_cells[i][0], fixed_cells[i][1], fixed_cells[i][2], fixed_cells[i][3])
            expression.append(clause)

            # Remove all the other possible values for this cell
            constrain_clause = Clause(size)
            for p in range(size):
                for q in range(size):
                    if p != fixed_cells[i][0] or q != fixed_cells[i][1]:
                        constrain_clause.add_negative(p, q, fixed_cells[i][2], fixed_cells[i][3])
            expression.append(constrain_clause)

    

    return expression


if __name__ == '__main__':
    expression = get_expression(3)
    for clause in expression:
        print(clause)
