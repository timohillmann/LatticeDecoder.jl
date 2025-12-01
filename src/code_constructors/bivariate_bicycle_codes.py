from multiprocessing import Value
from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
import numpy as np

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
    elif code_name == "18_4_3":
        # A: [(0, 0), (0, 1)]
        # B: [(0, 0), (1, 0), (2, 2)]
        # lx: 3
        # ly: 3
        a = 1 + x
        b = 1 + y + y**2 * x**2
        ell, m = 3, 3      
        return BivariateBicycle(a, b, ell, m, 1, code_name)  

    else:
        raise ValueError("Code not defined.")

PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/"
field = "p2"
# for code_string in ["72_12_6", "108_8_10", "144_12_12"]:
#     bb = ibm_codes(code_string)
#     np.savez(PATH + code_string + "_" + field,
#     hx=bb.hx, hz=bb.hz, lx=bb.x_logicals, lz=bb.z_logicals)

for code_string in ["18_4_3"]:
    # bb = nico_codes(code_string)
    bb = timos_code(code_string)
    np.savez(PATH + code_string + "_" + field,
    hx=bb.hx, hz=bb.hz, lx=bb.x_logicals, lz=bb.z_logicals)