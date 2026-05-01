from string import digits
from typing import Dict
from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
import numpy as np
import scipy.io as sio
import scipy.sparse as sparse
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


def parse_poly(poly_str, field):
    """
    Convert a string like '3 + 4x + 5xy^2' into
    Monomial(field) expressions.
    """
    
    x, y = Monomial(field, 'x'), Monomial(field, 'y')

    poly = 0

    # Replace exponent symbol
    poly_str = poly_str.replace("^", "**")

    # Normalize minus signs
    terms = poly_str.replace('-', '+-').split('+')

    for term in terms:
        term = term.strip()
        if not term:
            continue

        # split coefficient from monomial
        coef = ''
        mon = ''

        for i, ch in enumerate(term):
            if ch.isdigit() or (ch == '-' and i == 0):
                coef += ch
            else:
                mon = term[i:]
                break

        if coef == '' or coef == '-':  # e.g. "y", "-y"
            coef = coef + '1'

        coef = int(coef) % field.p

        # If no monomial part, just add constant
        if mon == '':
            poly += coef
            continue

        # Build monomial
        # Split into factors: x, y, x**2, y**3, x*y, x**2*y**4, etc.
        factors = []
        token = ""

        for ch in mon:
            if ch in ["x", "y"]:
                if token:
                    factors.append(token)
                token = ch
            else:
                token += ch
        if token:
            factors.append(token)

        # Evaluate factors
        m = 1
        for f in factors:
            m *= eval(f, {"x": x, "y": y})

        poly += coef * m

    return poly



def get_code_from_json_entry(entry:Dict, field, code_name=None):
    a = parse_poly(entry["a"], field)
    b = parse_poly(entry["b"], field)
    ell = entry["ell"]
    m   = entry["m"]

    if not code_name:
        code_name =  "_".join([str(param) for param in entry["parameters"]]) + f"_p{field.p}"


    code = BivariateBicycle(a, b, ell, m, 1, code_name)
    return code

codes = [{"parameters": [48, 2, 9], "a": "1 + 2x^3y^2", "b": "3 + 4x^2y^2 + 5x^5", "ell": 8, "m": 3}]

codes = [
    {"parameters": [24, 2, 5], "a": "1 + 2x^4y", "b": "3 + 4y + 5xy", "ell": 6, "m": 2},
    {"parameters": [36, 2, 7], "a": "1 + 2xy", "b": "3 + 4y^2 + 5y^5", "ell": 2, "m": 9},
    # {"parameters": [48, 2, 9], "a": "1 + 2x^3y^2", "b": "3 + 4x^2y^2 + 5x^5", "ell": 8, "m": 3},
    {"parameters": [60, 2, 11], "a": "1 + 2xy", "b": "3 + 4y + 5xy^4", "ell": 5, "m": 6},
    {"parameters": [84, 2, 13], "a": "1 + 2x^4y", "b": "3 + 4y^2 + 5x^2y^2", "ell": 7, "m": 6},
    {"parameters": [96, 2, 15], "a": "1 + 2xy^3", "b": "3 + 4x + 5x^4y", "ell": 6, "m": 8}
]


# p = 3
# codes= [
#     # {"parameters": [48, 4, 5], "a": "3y + 2y^2 + 3y^3", "b": "3y + 2y^5 + 3xy", "ell": 3, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [48, 4, 6], "a": "3y + 2y^2 + 3y^3", "b": "3y^2 + 2y^3 + 3xy^4", "ell": 3, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [64, 4, 6], "a": "3y + 2y^2 + 3y^3", "b": "3y^2 + 2y^3 + 3xy^4", "ell": 4, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [64, 8, 4], "a": "3y + 2y^2 + 3y^3", "b": "3y^2 + 2y^3 + 3x^2y^4", "ell": 4, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [80, 4, 6], "a": "3y + 2y^2 + 3y^3", "b": "3y^2 + 2y^3 + 3x^2y^4", "ell": 5, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [80, 4, 16], "a": "3y + 2y^2 + 3y^3", "b": "3y^2 + 2y^3 + 3xy^4", "ell": 5, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [96, 4, 6], "a": "3y + 2y^2 + 3y^3", "b": "3y + 2xy^2 + 3x^2y^4", "ell": 6, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [96, 8, 5], "a": "3y + 2y^2 + 3y^3", "b": "3y + 2y^5 + 3x^4y", "ell": 6, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [96, 8, 6], "a": "3y + 2y^2 + 3y^3", "b": "3y^2 + 2y^3 + 3x^2y^4", "ell": 6, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [96, 4, 12], "a": "3y + 2y^2 + 3y^3", "b": "3y^3 + 2x^2y^4 + 3x^3y", "ell": 6, "m": 8, "q": 7, "p": 7},
#     # {"parameters": [48, 2, 8], "a": "3y + 2y^2 + 3y^4", "b": "3y + 2y^2 + 3xy^4", "ell": 4, "m": 6, "q": 7, "p": 7}
#     # {"parameters": [54, 2, 9], "a": "3y + 2y^2 + 3y^4", "b": "3y^2 + 2y^6 + 3xy^5", "ell": 3, "m": 9, "q": 7, "p": 7},
#     # {"parameters": [72, 2, 9], "a": "3y + 2y^2 + 3y^4", "b": "3y^2 + 2xy^5 + 3x^2y", "ell": 4, "m": 9, "q": 7, "p": 7},
#     # {"parameters": [72, 4, 9], "a": "3y + 2y^2 + 3y^4", "b": "3y^2 + 2y^5 + 3x^2y^3", "ell": 4, "m": 9, "q": 7, "p": 7},
#     # {"parameters": [84, 4, 10], "a": "2 + 2x^2y^2 + 2x^5y", "b": "2 + 2y^2 + 2x^3y", "ell": 7, "m": 6, "q": 7, "p": 7},
#     # {"parameters": [96, 4, 9], "a": "2 + 2x^2y^2 + 2x^5y", "b": "2 + 2y + 2x^3y^2", "ell": 8, "m": 6, "q": 7, "p": 7},
#     # {"parameters": [84, 4, 12], "a": "2 + 2x^2y^2 + 2x^5y", "b": "2 + 2y^2 + 2x^2y", "ell": 7, "m": 6, "q": 7, "p": 7},
#     # {"parameters": [84, 2, 10], "a": "2 + 2x^3 + 2x^5y", "b": "2 + 2x^2y^4 + 2x^4y^2", "ell": 7, "m": 6, "q": 7, "p": 7},
#     # {"parameters": [54, 4, 8], "a": "2 + 2x^3y + 2x^5y", "b": "2 + 2x^5 + 2x^5y", "ell": 9, "m": 3, "q": 7, "p": 7},
#     # {"parameters": [48, 2, 6], "a": "2 + 2x^4 + 2x^5y", "b": "2 + 2x^2 + 2x^3y", "ell": 8, "m": 3, "q": 7, "p": 7}

#     # {"parameters": [24, 8, 2], "a": "1 + y^2 + y^4", "b": "1 + y + y^2", "ell": 2, "m": 6},
#     # {"parameters": [36, 8, 2], "a": "1 + y^2 + x^2y^2", "b": "1 + x + x^2", "ell": 3, "m": 6},
#     # {"parameters": [36, 8, 4], "a": "1 + y^2 + xy^2", "b": "1 + y^2 + x^2", "ell": 3, "m": 6},
#     # {"parameters": [48, 8, 4], "a": "1 + y^2 + x^2y", "b": "1 + x^2y^2 + x^4y", "ell": 8, "m": 3},
#     # {"parameters": [54, 8, 6], "a": "1 + y^2 + x^3y", "b": "1 + x^2y + x^4y^2", "ell": 9, "m": 3},
#     # {"parameters": [60, 8, 6], "a": "1 + y^2 + x^2y^4", "b": "1 + y^4 + xy^2", "ell": 5, "m": 6},
#     # {"parameters": [72, 8, 6], "a": "1 + y^2 + x^2y^2", "b": "1 + y^5 + x^4", "ell": 6, "m": 6},
#     # {"parameters": [96, 8, 8], "a": "1 + y^2 + x^2y", "b": "1 + x^2y^2 + x^4y", "ell": 8, "m": 6},
#     # {"parameters": [84, 8, 8], "a": "1 + y^2 + x^2y^4", "b": "1 + xy^4 + x^3y^2", "ell": 7, "m": 6},
#     # {"parameters": [108, 8, 10], "a": "1 + y^2 + x^3y", "b": "1 + xy^3 + x^5", "ell": 9, "m": 6},
#     # {"parameters": [120, 8, 10], "a": "1 + y^2 + x^2y^4", "b": "1 + xy^5 + x^5y", "ell": 10, "m": 6},
#     # {"parameters": [144, 8, 12], "a": "1 + y^2 + x^2y", "b": "1 + x^2y^4 + x^4y^2", "ell": 8, "m": 9},
#     # {"parameters": [162, 8, 12], "a": "1 + y^2 + x^3y", "b": "1 + x^2y^4 + x^4y^2", "ell": 9, "m": 9},
#     # {"parameters": [36, 4, 6], "a": "1 + y + x", "b": "1 + y + x^2y^4", "ell": 3, "m": 6},
#     # {"parameters": [54, 4, 8], "a": "1 + y + x", "b": "1 + y + x^2y^4", "ell": 3, "m": 9},
#     # # {"parameters": [72, 4, 8], "a": "1 + y + x", "b": "1 + y + x^2y^4", "ell": 6, "m": 6},
#     # {"parameters": [108, 4, 14], "a": "1 + y + x", "b": "1 + y + x^2y^4", "ell": 6, "m": 9},
#     # {"parameters": [108, 4, 16], "a": "1 + y + x", "b": "1 + xy^2 + x^4", "ell": 9, "m": 6},
#     # {"parameters": [48, 2, 6], "a": "1 + y + x", "b": "1 + y^2 + xy^3", "ell": 6, "m": 4},
#     # {"parameters": [60, 2, 8], "a": "1 + y + x", "b": "1 + y^2 + xy^3", "ell": 6, "m": 5},
#     # {"parameters": [60, 2, 10], "a": "1 + y + x", "b": "1 + y^2 + xy^4", "ell": 5, "m": 6},
#     # {"parameters": [72, 4, 8], "a": "1 + y + x", "b": "1 + y^2 + xy^3", "ell": 6, "m": 6},
#     # {"parameters": [84, 2, 10], "a": "1 + y + x", "b": "1 + y^2 + xy^3", "ell": 6, "m": 7},
#     # {"parameters": [96, 2, 12], "a": "1 + y + x", "b": "1 + y^2 + xy^3", "ell": 6, "m": 8},
#     # {"parameters": [84, 2, 12], "a": "1 + y + x", "b": "1 + y^2 + xy^4", "ell": 6, "m": 7},
#     # {"parameters": [120, 2, 14], "a": "1 + y + x", "b": "1 + y^2 + xy^5", "ell": 6, "m": 10},
#     # {"parameters": [162, 4, 18], "a": "1 + y + x", "b": "1 + y^4 + x^4", "ell": 9, "m": 9},
#     # {"parameters": [132, 2, 14], "a": "1 + y + x", "b": "1 + y + x^6", "ell": 11, "m": 6},
#     # {"parameters": [108, 6, 6], "a": "1 + y + y^2", "b": "1 + xy^5 + x^3y", "ell": 6, "m": 9}
#     # {"parameters": [176, 2, 12], "a": "1 + x^5y", "b": "1 + x^2y^3", "ell": 11, "m": 8},
#     # {"parameters": [128, 2, 8], "a": "1 + xy^3", "b": "1 + x^3y^2", "ell": 8, "m": 8},
#     # {"parameters": [100, 2, 10], "a": "1 + x^3y^2", "b": "1 + x^5y", "ell": 10, "m": 5},
#   {"parameters": [90, 6, 5], "a": "1 + y + y^2", "b": "1 + y + x^3", "ell": 9, "m": 5, "q": 3, "p": 3},
# {"parameters": [40, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 5, "m": 4, "q": 3, "p": 3},
# {"parameters": [50, 2, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 5, "m": 5, "q": 3, "p": 3},
# {"parameters": [60, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 5, "m": 6, "q": 3, "p": 3},
# {"parameters": [70, 2, 7], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 5, "m": 7, "q": 3, "p": 3},
# {"parameters": [80, 2, 8], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 5, "m": 8, "q": 3, "p": 3},
# {"parameters": [48, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 6, "m": 4, "q": 3, "p": 3},
# {"parameters": [60, 6, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 6, "m": 5, "q": 3, "p": 3},
# {"parameters": [72, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 6, "m": 6, "q": 3, "p": 3},
# {"parameters": [84, 6, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 6, "m": 7, "q": 3, "p": 3},
# {"parameters": [96, 6, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 6, "m": 8, "q": 3, "p": 3},
# {"parameters": [56, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 7, "m": 4, "q": 3, "p": 3},
# {"parameters": [70, 2, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 7, "m": 5, "q": 3, "p": 3},
# {"parameters": [84, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 7, "m": 6, "q": 3, "p": 3},
# {"parameters": [98, 2, 7], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 7, "m": 7, "q": 3, "p": 3},
# {"parameters": [64, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 8, "m": 4, "q": 3, "p": 3},
# {"parameters": [80, 2, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 8, "m": 5, "q": 3, "p": 3},
# {"parameters": [96, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 8, "m": 6, "q": 3, "p": 3},
# {"parameters": [72, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 9, "m": 4, "q": 3, "p": 3},
# {"parameters": [90, 6, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y", "ell": 9, "m": 5, "q": 3, "p": 3},
# {"parameters": [40, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 5, "m": 4, "q": 3, "p": 3},
# {"parameters": [50, 2, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 5, "m": 5, "q": 3, "p": 3},
# {"parameters": [60, 4, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 5, "m": 6, "q": 3, "p": 3},
# {"parameters": [70, 2, 7], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 5, "m": 7, "q": 3, "p": 3},
# {"parameters": [80, 2, 8], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 5, "m": 8, "q": 3, "p": 3},
# {"parameters": [48, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 6, "m": 4, "q": 3, "p": 3},
# {"parameters": [60, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 6, "m": 5, "q": 3, "p": 3},
# {"parameters": [72, 12, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 6, "m": 6, "q": 3, "p": 3},
# {"parameters": [84, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 6, "m": 7, "q": 3, "p": 3},
# {"parameters": [96, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 6, "m": 8, "q": 3, "p": 3},
# {"parameters": [56, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 7, "m": 4, "q": 3, "p": 3},
# {"parameters": [70, 2, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 7, "m": 5, "q": 3, "p": 3},
# {"parameters": [84, 4, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 7, "m": 6, "q": 3, "p": 3},
# {"parameters": [98, 2, 7], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 7, "m": 7, "q": 3, "p": 3},
# {"parameters": [64, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 8, "m": 4, "q": 3, "p": 3},
# {"parameters": [80, 2, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 8, "m": 5, "q": 3, "p": 3},
# {"parameters": [96, 4, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 8, "m": 6, "q": 3, "p": 3},
# {"parameters": [72, 6, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 9, "m": 4, "q": 3, "p": 3},
# {"parameters": [90, 6, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y^2", "ell": 9, "m": 5, "q": 3, "p": 3},
# {"parameters": [50, 2, 5], "a": "1 + y + y^2", "b": "1 + y + x^3y^3", "ell": 5, "m": 5, "q": 3, "p": 3},
# {"parameters": [60, 2, 4], "a": "1 + y + y^2", "b": "1 + y + x^3y^3", "ell": 5, "m": 6, "q": 3, "p": 3},
# {"parameters": [70, 2, 7], "a": "1 + y + y^2", "b": "1 + y + x^3y^3", "ell": 5, "m": 7, "q": 3, "p": 3},
# {"parameters": [80, 2, 8], "a": "1 + y + y^2", "b": "1 + y + x^3y^3", "ell": 5, "m": 8, "q": 3, "p": 3},
# {"parameters": [90, 2, 6], "a": "1 + y + y^2", "b": "1 + y + x^3y^3", "ell": 5, "m": 9, "q": 3, "p": 3},
# ]

p = 2
field = Field(p)
from collections import defaultdict
name_mult = defaultdict(int)
import os
import json
PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/weight_5_p2/"

with open("/Users/timo/Documents/LatticeDecoder.jl/analyzed_codes_weight_5_p2.json", "r") as f:
    lines = f.readlines()

codes = [json.loads(line) for line in lines]

os.makedirs(PATH, exist_ok=True)
for code in codes:
    if code["parameters"][0] < 100:
        bb = get_code_from_json_entry(code, field)
        np.savez(PATH + bb.name + f"_{name_mult[bb.name]}",
        hx=bb.hx, hz=bb.hz, lx=bb.x_logicals, lz=bb.z_logicals)
        name_mult[bb.name] += 1
    else:
        print(code["parameters"])