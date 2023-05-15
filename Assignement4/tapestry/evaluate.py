# File to automate the evalutation of some propositional logic formulas

def sign_not(A):
    if A == 1: return 0
    else: return 1

def sign_and(A, B):
    if A == 1 and B == 1: return 1
    else: return 0

def sign_or(A, B):
    if A == 1 or B == 1: return 1
    else: return 0

def sign_implies(A, B):
    if A == 1 and B == 0: return 0
    else: return 1


# Evaluate formula 1
def evaluate_formulas(d=False):
    # ¬(A ∧ B) ∨ (¬B ∧ C )
    formula = lambda a, b, c:  sign_or(sign_not(sign_and(a, b)), sign_and(sign_not(b), c))

    # (¬A ∨ B) ⇒ C
    # formula = lambda a, b, c: sign_implies(sign_or(sign_not(a), b), c)

    # (A ∨ ¬B) ∧ (¬B ⇒ ¬C ) ∧ ¬(D ⇒ ¬A)
    # formula = lambda a, b, c, d: sign_and(sign_and(sign_or(a, sign_not(b)), sign_implies(sign_not(b), sign_not(c))), sign_not(sign_implies(d, sign_not(a))))

    # Define variables
    A = (0, 1)
    B = (0, 1)
    C = (0, 1)
    D = (0, 1)  
    valid = 0

    for i in range(2):
        for j in range(2):
            for k in range(2):
                if d:
                    for l in range(2):
                        result = formula(A[i], B[j], C[k], D[l])
                        if result == 1: valid += 1
                        print("A: ", A[i], " B: ", B[j], " C: ", C[k], " D: ", D[l], " Result: ", result)
                else:
                    result = formula(A[i], B[j], C[k])
                    if result == 1: valid += 1
                    print("A: ", A[i], " B: ", B[j], " C: ", C[k], " Result: ", result)
    print("Valid: ", valid)


# Launch
evaluate_formulas()

# Results for first formula
#   A:  0  B:  0  C:  0  Result:  1
#   A:  0  B:  0  C:  1  Result:  1
#   A:  0  B:  1  C:  0  Result:  1
#   A:  0  B:  1  C:  1  Result:  1
#   A:  1  B:  0  C:  0  Result:  1
#   A:  1  B:  0  C:  1  Result:  1
#   A:  1  B:  1  C:  0  Result:  0
#   A:  1  B:  1  C:  1  Result:  0
#   Valid:  6


# Results for second formula
#   A:  0  B:  0  C:  0  Result:  0
#   A:  0  B:  0  C:  1  Result:  1
#   A:  0  B:  1  C:  0  Result:  0
#   A:  0  B:  1  C:  1  Result:  1
#   A:  1  B:  0  C:  0  Result:  1
#   A:  1  B:  0  C:  1  Result:  1
#   A:  1  B:  1  C:  0  Result:  0
#   A:  1  B:  1  C:  1  Result:  1
#   Valid:  5


# Results for third formula
#   A:  0  B:  0  C:  0  D:  0  Result:  0
#   A:  0  B:  0  C:  0  D:  1  Result:  0
#   A:  0  B:  0  C:  1  D:  0  Result:  0
#   A:  0  B:  0  C:  1  D:  1  Result:  0
#   A:  0  B:  1  C:  0  D:  0  Result:  0
#   A:  0  B:  1  C:  0  D:  1  Result:  0
#   A:  0  B:  1  C:  1  D:  0  Result:  0
#   A:  0  B:  1  C:  1  D:  1  Result:  0
#   A:  1  B:  0  C:  0  D:  0  Result:  0
#   A:  1  B:  0  C:  0  D:  1  Result:  1
#   A:  1  B:  0  C:  1  D:  0  Result:  0
#   A:  1  B:  0  C:  1  D:  1  Result:  0
#   A:  1  B:  1  C:  0  D:  0  Result:  0
#   A:  1  B:  1  C:  0  D:  1  Result:  1
#   A:  1  B:  1  C:  1  D:  0  Result:  0
#   A:  1  B:  1  C:  1  D:  1  Result:  1
#   Valid:  3
