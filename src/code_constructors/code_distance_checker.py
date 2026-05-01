from string import digits
from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
import numpy as np
import json
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

import json

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


def run_qdistrnd(code, codename, field:int):
    # path = f"./_codes/{encode_poly(code.poly_A)}_{encode_poly(code.poly_B)}/{codename}/"
    path = f"/Users/timo/Documents/LatticeDecoder.jl/data/qudit_codes/.tmp/"
    names = ["hx", "hz"]
    hx = code.hx
    hz = code.hz
    os.makedirs(path, exist_ok=True)
    for mat, name in zip([hx, hz], names):
        if type(mat) != type(None):
            # np.savetxt(path + name + ".txt", mat, fmt="%i")
            sio.mmwrite(
                path + name + ".mtx",
                sparse.coo_matrix(mat),
                comment=f"Field: GF({field})",
            )


    subprocess.run(["bash", "/Users/timo/Documents/LatticeDecoder.jl/data/compute_distance.sh", path, str(1000), str(field)])

    # rename the folder to include the distance
    time.sleep(0.1)
    with open(f"{path}info.txt", "r") as f:
        lines = f.readlines()

        # Extract dZ and dX from the info.txt file
        for line in lines:
            line = line.strip()
            if line.startswith("dZ="):
                dZ = int(line.split("=")[1].strip())
            elif line.startswith("dX="):
                dX = int(line.split("=")[1].strip())

    return dX, dZ





# def run_qdistrnd(code, codename, field:int):
#     # path = f"./_codes/{encode_poly(code.poly_A)}_{encode_poly(code.poly_B)}/{codename}/"
#     path = f"/Users/timo/Documents/LatticeDecoder.jl/data/qudit_codes/{codename}/"
#     names = ["hx", "hz"]
#     hx = code.hx
#     hz = code.hz
#     os.makedirs(path, exist_ok=True)
#     for mat, name in zip([hx, hz], names):
#         if type(mat) != type(None):
#             # np.savetxt(path + name + ".txt", mat, fmt="%i")
#             sio.mmwrite(
#                 path + name + ".mtx",
#                 sparse.coo_matrix(mat),
#                 comment=f"Field: GF({field})",
#             )

#         with open(f"{path}params.txt", "w") as f:
#             f.write(f"a = {code.a}\n")
#             f.write(f"b = {code.b}\n")
#             # f.write(f"lx: {lx}\n")
#             # f.write(f"ly: {ly}\n")
#             f.flush()
#     os.makedirs(path, exist_ok=True)

#     subprocess.run(["bash", "/Users/timo/Documents/LatticeDecoder.jl/data/compute_distance.sh", path, str(10000), str(field)])

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

#     new_path = path.split("_")[0] + f"_{code.parameters[0]}" + f"_{code.parameters[1]}" + f"_{min(dX, dZ)}"
#     flag = True
#     idx = 0
#     while flag:
#         try:
#             flag = False
            
#             os.rename(path, new_path + f"_{idx}")
#         except:
#             idx += 1
            






import json
import time
import os
# found_codes_weight_6_p7_323323
w = 6
p = 3
field = Field(p)
OFFSET_FILE = f"/Users/timo/Documents/LatticeDecoder.jl/found_codes_weight_{w}_p{p}.offset"
INPUT_FILE  = f"/Users/timo/Documents/LatticeDecoder.jl/found_codes_weight_{w}_p{p}.json"
OUTPUT_FILE = f"/Users/timo/Documents/LatticeDecoder.jl/analyzed_codes_weight_{w}_p{p}.json"

def load_offset():
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, "r") as f:
            return int(f.read())
    return 0

def save_offset(offset):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))

offset = load_offset()

with open(INPUT_FILE, "r") as f:
    f.seek(offset)

    while True:
        pos = f.tell()
        line = f.readline()

        if not line:
            time.sleep(0.5)
            continue

        # Handle partial writes safely
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # writer hasn't finished this line yet
            f.seek(pos)
            time.sleep(0.5)
            continue

        # ---- YOUR EXISTING LOGIC ----
        a = parse_poly(entry["a"], field)
        b = parse_poly(entry["b"], field)
        ell = entry["ell"]
        m   = entry["m"]

        code = BivariateBicycle(a, b, ell, m, 1, "code_string")
        dX, dZ = run_qdistrnd(code, "code", p)

        entry["parameters"][-1] = min(dX, dZ)

        with open(OUTPUT_FILE, "a") as out:
            out.write(json.dumps(entry) + "\n")

        # --------------------------------

        # Update offset *after* successful processing
        offset = f.tell()
        save_offset(offset)



# with open(path, "r") as f:
#     # Go to end of file
#     f.seek(0, 2)

#     while True:
#         line = f.readline()
#         if not line:
#             time.sleep(0.5)  # wait for new data
#             continue

#         entry = json.loads(line)

#         # ---- process new record ----
#         a = parse_poly(entry["a"], field)
#         b = parse_poly(entry["b"], field)
#         ell = entry["ell"]
#         m   = entry["m"]

#         code = BivariateBicycle(a, b, ell, m, 1, "code_string")
#         dX, dZ = run_qdistrnd(code, "code", p)

#         entry["parameters"][-1] = min(dX, dZ)

#         with open(f"analyzed_codes_w{w}_p{p}.json", "a") as out:
#             out.write(json.dumps(entry) + "\n")



# w = 6
# records = []
# with open(f"/Users/timo/Documents/LatticeDecoder.jl/found_codes_weight_{w}.json") as f:
#     for line in f:
#         records.append(json.loads(line))

# RECS = records


# for idx, entry in enumerate(RECS):
#     print(f"{idx} / {len(RECS)}", end="\r")
#     a = parse_poly(entry["a"], field)
#     b = parse_poly(entry["b"], field)
#     ell = entry["ell"]
#     m   = entry["m"]

#     code = BivariateBicycle(a, b, ell, m, 1, "code_string")
#     dX, dZ = run_qdistrnd(code, f"code_{idx}", p)

#     # new_record = entry.copy()
#     entry["parameters"][-1] = min(dX, dZ)
#     # new_record["dX"] = dX
#     # new_record["dZ"] = dZ
#     with open(f"analyzed_codes_weight_{w}.json", "a") as f:
#         f.write(json.dumps(entry) + "\n")

