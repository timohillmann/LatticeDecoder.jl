from string import digits
from typing import Dict
from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.qudit_code import QuditCode
import numpy as np
import json

def get_code_f3(code_string: str):

    if code_string == "24_4_4":
        field = Field(3)
        x, y = Monomial(field, 'x'), Monomial(field, 'y')
        a = x + x**2
        b = x**3 + 2 * y + 2 * y**2
        ell, m = 4, 3
    elif code_string == "30_4_5":
        field = Field(3)
        x, y = Monomial(field, 'x'), Monomial(field, 'y')
        a = 1 + y + x**2 * y**2
        b = 2 + x
        ell, m = 5, 3
    elif code_string == "48_4_7":
        field = Field(3)
        x, y = Monomial(field, 'x'), Monomial(field, 'y')
        a = 1 + 2*x
        b = 1 + y + x**3 * y**2
        ell, m = 8, 3
    else:
        raise ValueError("`code_string` not recognized.")
    
    return BivariateBicycle(a, b, ell,  m, 1, code_string)

 
def get_code_f5(code_string: str):

    if code_string == "30_4_5":
        field = Field(5)
        x, y = Monomial(field, 'x'), Monomial(field, 'y')
        a = 1 + y + x**2 * y**2
        b = 2 + 3 * x
        ell, m = 5, 3
    elif code_string == "48_4_7":
        field = Field(5)
        x, y = Monomial(field, 'x'), Monomial(field, 'y')
        a = 2 + 4*x
        b = 3 + 3*y + x**3 * y**2
        ell, m = 8, 3
    else:
        raise ValueError("`code_string` not recognized.")
    
    return BivariateBicycle(a, b, ell,  m, 1, code_string)

       
 
def get_code_f7(code_string: str):

    if code_string == "30_4_5":
        field = Field(7)
        x, y = Monomial(field, 'x'), Monomial(field, 'y')
        a = 4*x + 2*y + x**2
        b = y**4 + 2*y**3 + 4*y**2
        ell, m = 5, 3
    elif code_string == "48_4_7":
        field = Field(7)
        x, y = Monomial(field, 'x'), Monomial(field, 'y')
        a = 1 + 3*x + y**4
        b = 4 + 6*y*x + 2 * x**4 * y**5
        ell, m = 8, 3
    else:
        raise ValueError("`code_string` not recognized.")
    
    return BivariateBicycle(a, b, ell,  m, 1, code_string)

       
def ibm_codes(code_name):
    field = Field(2)
    x, y = Monomial(field, 'x'), Monomial(field, 'y')
    if code_name == "72_12_6":
        a = y**3 + x + x**2
        b = x**3 + y + y**2
        ell, m = 6, 6
        return BivariateBicycle(a, b, ell, m, 1, code_name)
    elif code_name == "108_8_10":
        a = y**3 + x + x**2
        b = x**3 + y + y**2
        ell, m = 9, 6
        return BivariateBicycle(a, b, ell, m, 1, code_name)
    elif code_name == "144_12_12":
        a = y**3 + x + x**2
        b = x**3 + y + y**2
        ell, m = 12, 6
        return BivariateBicycle(a, b, ell, m, 1, code_name)
    else:
        raise ValueError("Code not defined.")
        
def nico_codes(code_name):
    field = Field(2)
    x, y = Monomial(field, 'x'), Monomial(field, 'y')
    if code_name == "30_4_5":
        a = 1 + x
        b = 1 + y + x**2 * y**2
        ell, m = 5, 3
        return BivariateBicycle(a, b, ell, m, 1, code_name)
    elif code_name == "48_4_7":
        a = 1 + x
        b = 1 + y + x**3 * y**2
        ell, m = 8, 3
        return BivariateBicycle(a, b, ell, m, 1, code_name)
    else:
        raise ValueError("Code not defined.")


def timos_code(code_name):
    # A: [(0, 0), (0, 2)]
    # B: [(0, 0), (1, 0), (2, 3)]
    # lx: 13
    # ly: 3
    field = Field(2)
    x, y = Monomial(field, 'x'), Monomial(field, 'y')
    if code_name == "78_4_9":
        a = 1 + x**2
        b = 1 + y + y**2 * x**3
        ell, m = 13, 3
        # code_name = "78_4_9"
        return BivariateBicycle(a, b, ell, m, 1, code_name)
    # elif code_name == "18_4_3":
    #     # A: [(0, 0), (0, 1)]
    #     # B: [(0, 0), (1, 0), (2, 2)]
    #     # lx: 3
    #     # ly: 3
    #     a = 1 + x
    #     b = 1 + y + y**2 * x**2
    #     ell, m = 3, 3      
    #     return BivariateBicycle(a, b, ell, m, 1, code_name)  

    elif code_name == "18_4_3":
        a = 1 + x**1
        b = 1 + y**2 + y**1 * x**1
        ell, m =  15, 3
        return BivariateBicycle(a, b, ell, m, 1, code_name)  

def surface_code(L: int):
    field = Field(2)
    x, y = Monomial(field, 'x'), Monomial(field, 'y')
    a = 1 + x
    b = 1 + y
    ell, m = L, L
    return BivariateBicycle(a, b, ell, m, 1, f"{2*L**2}_2_{L}_p2")


# PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/"
# field = "p2"
# # for code_string in ["72_12_6", "108_8_10", "144_12_12"]:
# #     bb = ibm_codes(code_string)
# #     np.savez(PATH + code_string + "_" + field,
# #     hx=bb.hx, hz=bb.hz, lx=bb.x_logicals, lz=bb.z_logicals)

# for code_string in ["90_4_9"]:
#     # bb = nico_codes(code_string)
#     bb = timos_code(code_string)
#     np.savez(PATH + code_string + "_" + field,
#     hx=bb.hx, hz=bb.hz, lx=bb.x_logicals, lz=bb.z_logicals)

import os
import scipy.io as sio
import scipy.sparse as sparse
import os
import subprocess
import time
import scipy.io as sio
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

# p = 7
# field = Field(p)
# x, y = Monomial(field, 'x'), Monomial(field, 'y')
# a = 1 + 2*x
# b = 3*x**2 + 4*y + 5*x**2 * y**2
# ell, m = 7, 3
# code = BivariateBicycle(a, b, ell,  m, 1, "code_string")


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




# def run_qdistrnd(codename, hx, hz, p:int, num_info_sets=2000):


#     # path = f"./_codes/{encode_poly(code.poly_A)}_{encode_poly(code.poly_B)}/{codename}/"
#     path = f"data/qudit_codes/{codename}/"
#     names = ["hx", "hz"]
#     os.makedirs(path, exist_ok=True)
#     for mat, name in zip([hx, hz], names):
#         if type(mat) != type(None):
#             # np.savetxt(path + name + ".txt", mat, fmt="%i")
#             sio.mmwrite(
#                 path + name + ".mtx",
#                 sparse.coo_matrix(mat),
#                 comment=f"Field: GF({p})",
#             )


#     os.makedirs(path, exist_ok=True)

#     subprocess.run(["bash", "/Users/timo/Documents/LatticeDecoder.jl/data/compute_distance.sh", path, str(num_info_sets)])

#     # rename the folder to include the distance
#     time.sleep(0.1)
#     with open(f"{path}info.txt", "r") as f:
#         lines = f.readlines()

#         # Extract dZ and dX from the info.txt file
#         for line in lines:
#             line = line.strip()
#             if line.startswith("dZ="):
#                 dZ = int(line.split("=")[1].strip())
#             elif line.startswith("dX="):
#                 dX = int(line.split("=")[1].strip())

#     new_path = path.replace("nan", f"{min(dX, dZ):d}")
#     os.rename(path, new_path)


# # run_qdistrnd("test_code_1", code.hx, code.hz, p)
# if code.parameters[1] != 0:
#     hx = code.hx
#     hz = code.hz
#     codename = "test_code_1"
#     # path = f"./_codes/{encode_poly(code.poly_A)}_{encode_poly(code.poly_B)}/{codename}/"
#     path = f"/Users/timo/Documents/LatticeDecoder.jl/data/qudit_codes/{codename}/"
#     names = ["hx", "hz"]
#     os.makedirs(path, exist_ok=True)
#     for mat, name in zip([hx, hz], names):
#         if type(mat) != type(None):
#             # np.savetxt(path + name + ".txt", mat, fmt="%i")
#             sio.mmwrite(
#                 path + name + ".mtx",
#                 sparse.coo_matrix(mat),
#                 comment=f"Field: GF({p})",
#             )

#         with open(f"{path}params.txt", "w") as f:
#             f.write(f"a = {code.a}\n")
#             f.write(f"b = {code.b}\n")
#             # f.write(f"lx: {lx}\n")
#             # f.write(f"ly: {ly}\n")
#             f.flush()
#     os.makedirs(path, exist_ok=True)

#     subprocess.run(["bash", "/Users/timo/Documents/LatticeDecoder.jl/data/compute_distance.sh", path, str(10000), str(p)])

#     # rename the folder to include the distance
#     time.sleep(0.1)
#     with open(f"{path}info.txt", "r") as f:
#         lines = f.readlines()

#         # Extract dZ and dX from the info.txt file
#         for line in lines:
#             line = line.strip()
#             if line.startswith("dZ="):
#                 dZ = int(line.split("=")[1].strip())
#             elif line.startswith("dX="):
#                 dX = int(line.split("=")[1].strip())

#     new_path = path.replace("nan", f"{min(dX, dZ):d}")
#     os.rename(path, new_path)




def generate_monomials(x, y, max_degree):
    monomials = []
    for dx in range(max_degree + 1):
        for dy in range(max_degree + 1):
            if 0 < dx + dy <= max_degree:
                monomials.append(x**dx * y**dy)
    return monomials


from itertools import combinations

def make_all_a_b(monomials):
    A_list = []
    B_list = []

    # a has 2 terms → use coefficients 1, 2
    for mono in combinations(monomials, 2):
        a = 1 + 1 * mono[0] + 1 * mono[1]
        A_list.append(a)

    # b has 3 terms → use coefficients 3, 4, 5
    for mono in combinations(monomials, 2):
        b = 1 + 1 * mono[0] + 1 * mono[1]
        B_list.append(b)

    return A_list, B_list


def sweep_codes(A_list, B_list, x, y, field, ell_offset=5, m_offset=5):
    codes = []
    len_A = len(A_list)
    len_B = len(B_list)
    digits_A = int(np.floor(np.log10(len_A)) + 1)
    digits_B = int(np.floor(np.log10(len_B)) + 1)
    for idxa, a in enumerate(A_list):
        for idx, b in enumerate(B_list):
            print(a, " ", b)
            print(f"A: {idxa:<{digits_A}}/{len_A}, B: {idx:<{digits_B}}/{len_B}", end="\r")
            deg_a, deg_b = ((a + b).coefficients).shape

            ell = deg_a + ell_offset
            m   = deg_b + m_offset
            for ell in range(deg_a + 1, deg_a + ell_offset + 1):
                for m in range(deg_b + 1, deg_b + m_offset + 1):
                    if (2 * m*ell) < 100:
                        code = BivariateBicycle(a, b, ell, m, 1, "code_string")
                        if code.parameters[1] != 0:
                            print("Found code.")
                            entry = {
                                "parameters": list(code.parameters),
                                "a": str(code.a),
                                "b": str(code.b),
                                "ell": code.l,
                                "m": code.m,
                                "q": field.p,
                                "p": field.p,
                            }
                            print(code.parameters)
                            with open("found_codes_weight_6_p3.json", "a") as f:
                                f.write(json.dumps(entry) + "\n")

                # codes.append(code)

    return codes


def get_code_from_json_entry(entry:Dict, field, code_name=None):
    a = parse_poly(entry["a"], field)
    b = parse_poly(entry["b"], field)
    ell = entry["ell"]
    m   = entry["m"]

    if not code_name:
        code_name =  "_".join([str(param) for param in entry["parameters"]]) + f"_p{field.p}"


    code = BivariateBicycle(a, b, ell, m, 1, code_name)
    return code

# codes = [{"parameters": [48, 2, 9], "a": "1 + 2x^3y^2", "b": "3 + 4x^2y^2 + 5x^5", "ell": 8, "m": 3}]

# codes = [
#     {"parameters": [24, 2, 5], "a": "1 + 2x^4y", "b": "3 + 4y + 5xy", "ell": 6, "m": 2},
#     {"parameters": [36, 2, 7], "a": "1 + 2xy", "b": "3 + 4y^2 + 5y^5", "ell": 2, "m": 9},
#     # {"parameters": [48, 2, 9], "a": "1 + 2x^3y^2", "b": "3 + 4x^2y^2 + 5x^5", "ell": 8, "m": 3},
#     {"parameters": [60, 2, 11], "a": "1 + 2xy", "b": "3 + 4y + 5xy^4", "ell": 5, "m": 6},
#     {"parameters": [84, 2, 13], "a": "1 + 2x^4y", "b": "3 + 4y^2 + 5x^2y^2", "ell": 7, "m": 6},
#     {"parameters": [96, 2, 15], "a": "1 + 2xy^3", "b": "3 + 4x + 5x^4y", "ell": 6, "m": 8}
# ]

# PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/weight_5/"
# p = 2
# codes= [
# {"parameters": [42, 6, 3], "a": "1 + y", "b": "1 + x + x^3y", "ell": 7, "m": 3, "q": 2, "p": 2},
# {"parameters": [56, 6, 4], "a": "1 + y", "b": "1 + x + x^3y", "ell": 7, "m": 4, "q": 2, "p": 2},
# {"parameters": [70, 6, 5], "a": "1 + y", "b": "1 + x + x^3y", "ell": 7, "m": 5, "q": 2, "p": 2},
# {"parameters": [84, 6, 6], "a": "1 + y", "b": "1 + x + x^3y", "ell": 7, "m": 6, "q": 2, "p": 2},
# {"parameters": [98, 6, 7], "a": "1 + y", "b": "1 + x + x^3y", "ell": 7, "m": 7, "q": 2, "p": 2},
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
# ]

# field = Field(p)
# for code in codes:
#     bb = get_code_from_json_entry(code, field)
#     np.savez(PATH + bb.name,
#     hx=bb.hx, hz=bb.hz, lx=bb.x_logicals, lz=bb.z_logicals)




# p = 5
# field = Field(p)
# x, y = Monomial(field, 'x'), Monomial(field, 'y')



# a = 1 + 2 * x**3 + 4 * x**4 + x**5


# def symplectic_matrix(n):
#     id = np.identity(n//2, dtype=int)
#     zero = np.zeros_like(id)
#     top = np.hstack((zero,id))
#     bottom = np.hstack((-id, zero))
#     return np.vstack((top, bottom))




# l = 10
# m = 1
# bb = BivariateBicycle(a, b, l, m, 1)
# bb.parameters

for L in [3, 4, 5, 6, 7, 8, 9, 10]:
    bb = surface_code(L)
    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/surface_codes/"
    np.savez(PATH + bb.name,
        hx=bb.hx, hz=bb.hz, lx=bb.x_logicals, lz=bb.z_logicals)


# max_degree = 6
# monomials = generate_monomials(x, y, max_degree)

# A_list, B_list = make_all_a_b(monomials)
# print(len(A_list))
# print(len(B_list))
# print(len(A_list)*len(B_list))
# codes = sweep_codes(A_list, B_list, x, y, field)
