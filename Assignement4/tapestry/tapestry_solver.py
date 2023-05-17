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

# python3 solve_linux.py instances/i01.txt

def get_expression(size, fixed_cells=None):

    expression = []

    # A Clause is a set of literals linked by an OR
    # All the clauses are linked by an AND

    # Add the fixed cells
    if fixed_cells is not None:
        for i in range(len(fixed_cells)):
            fixed_clause = Clause(size)
            fixed_clause.add_positive(fixed_cells[i][0], fixed_cells[i][1], fixed_cells[i][2], fixed_cells[i][3])
            expression.append(fixed_clause)

            # This must not be adde (because it is take into account in the other loops)
            # But its speedup calculus a little bit
            constrain_clause = Clause(size)
            for p in range(size):
                for q in range(size):
                    if p != fixed_cells[i][0] or q != fixed_cells[i][1]:
                        constrain_clause.add_negative(p, q, fixed_cells[i][2], fixed_cells[i][3])
            expression.append(constrain_clause)

    # There is only one shape and one color per cell : !C(ijab) or !C(ija2b2) for all a2, b2 != a, b
    # Each pair shape/color must be unique : !C(ijab) or !C(ijab), i, j != i2, j2
    for i in range(size):
        for j in range(size):
            for a in range(size):
                for b in range(size):
                    for a2 in range(size):
                        for b2 in range(size):
                            if a != a2 or b != b2:
                                # Only one per cell
                                single_clause = Clause(size)
                                single_clause.add_negative(i, j, a, b)
                                single_clause.add_negative(i, j, a2, b2)
                                expression.append(single_clause) 
                            
                                # It must be unique   
                                unique_clause = Clause(size)
                                unique_clause.add_negative(a, b, i, j)
                                unique_clause.add_negative(a2, b2, i, j)
                                expression.append(unique_clause)                                          

    # Each shape and each color must be found in each line 
    # Each shape and each color must be found in each column
    for i in range(size):       # For one line or column
        for p in range(size):   # For one shape or color
            line_shape_clause = Clause(size)
            line_color_clause = Clause(size)
            column_shape_clause = Clause(size)
            column_color_clause = Clause(size)
            for j in range(size):      # For all the other lines or columns
                for q in range(size):  # For all the other lines or columns
                    line_shape_clause.add_positive(i, j, p, q)
                    line_color_clause.add_positive(i, j, q, p)
                    column_shape_clause.add_positive(j, i, p, q)
                    column_color_clause.add_positive(j, i, q, p)
            expression.append(line_shape_clause)
            expression.append(line_color_clause)
            expression.append(column_shape_clause)
            expression.append(column_color_clause)

    return expression


if __name__ == '__main__':
    expression = get_expression(3)
    for clause in expression:
        print(clause)
