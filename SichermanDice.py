"""
Sicherman Dice via Higher-Order Unconstrained Binary Optimization (HUBO)
=========================================================================

Goal
----
Find two 6-sided dice with positive integer faces (not necessarily 1-6)
whose sum distribution over all 36 (face_A, face_B) outcomes exactly
matches the sum distribution of two standard dice:

    sum:    2  3  4  5  6  7  8  9 10 11 12
    count:  1  2  3  4  5  6  5  4  3  2  1

Encoding
--------
For die A, face i (i = 0..5) and candidate value v (v = 1..VmaxA):
    x[i][v] = 1  if face i of die A equals v, else 0
Similarly y[j][w] for die B, face j, value w (w = 1..VmaxB).

Structural constraint (NEW)
----------------------------
Die A is restricted to a small max face value (VmaxA = 4). This is the
known structure of the non-trivial Sicherman solution {1,2,2,3,3,4}.
Restricting A's domain does two important things:
  1. It makes the *trivial* {1..6}/{1..6} solution structurally
     impossible to represent (6 is not in A's domain), so any
     zero-energy solution the annealer finds MUST be the non-trivial
     (Sicherman) solution.
  2. It shrinks the search space dramatically (4^6 possibilities for A
     instead of up to Vmax^6), which makes it far easier for simulated
     annealing to actually reach the true global optimum instead of
     stalling near it.

Die B is left "unlimited" up to VmaxB = 11 -- the largest a face could
possibly need to be, since with A's minimum face value of 1, reaching
a sum of 12 requires a B face of 11.

Define:
    c_v = sum_i x[i][v]      (# of A-faces equal to v)
    d_w = sum_j y[j][w]      (# of B-faces equal to w)

Objective (HUBO, degree up to 4):
    H = sum_s ( sum_{v+w=s} c_v * d_w  -  target[s] )^2
        + penalty * sum_i ( sum_v x[i][v] - 1 )^2      (one-hot per A-face)
        + penalty * sum_j ( sum_w y[j][w] - 1 )^2      (one-hot per B-face)

Note: since x,y are binary, x^2 = x, so whenever a monomial contains a
repeated variable we collapse it (handled in add_term via set-dedup).
"""

import itertools
import openjij as oj

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
n_faces = 6              # each die has 6 faces

VmaxA = 4                # die A restricted to values 1..4 (Sicherman structure)
VmaxB = 11                # die B "unlimited" -- large enough to cover any needed sum

valuesA = list(range(1, VmaxA + 1))
valuesB = list(range(1, VmaxB + 1))

penalty = 150.0           # one-hot constraint weight; must dominate objective scale
num_reads = 500
num_sweeps = 8000

# ----------------------------------------------------------------------
# Target distribution: sums of two standard d6 dice
# ----------------------------------------------------------------------
target = {}
for a in range(1, 7):
    for b in range(1, 7):
        s = a + b
        target[s] = target.get(s, 0) + 1
# target = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1}

# ----------------------------------------------------------------------
# Variable naming helpers
# ----------------------------------------------------------------------
def xA(i, v):
    return f"A_{i}_{v}"

def xB(j, w):
    return f"B_{j}_{w}"

# ----------------------------------------------------------------------
# HUBO dictionary construction
# ----------------------------------------------------------------------
hubo = {}

def add_term(term_vars, coeff):
    """Add coeff to the monomial defined by term_vars (dedup, since x^2=x)."""
    key = tuple(sorted(set(term_vars)))
    if not key:
        return  # ignore pure constants
    hubo[key] = hubo.get(key, 0.0) + coeff

# --- 1) Sum-distribution matching term (degree up to 4) ---
s_min, s_max = 2, VmaxA + VmaxB
for s in range(s_min, s_max + 1):
    t = target.get(s, 0)

    # sum_{v+w=s} c_v*d_w  expanded into elementary (x_{i,v}, y_{j,w}) products
    inner_terms = []
    for v in valuesA:
        w = s - v
        if w in valuesB:
            for i in range(n_faces):
                for j in range(n_faces):
                    inner_terms.append((xA(i, v), xB(j, w)))

    # (sum_k T_k - t)^2 = sum_k sum_l T_k*T_l  - 2t * sum_k T_k  + t^2 (const, dropped)
    for (xa1, xb1) in inner_terms:
        for (xa2, xb2) in inner_terms:
            add_term([xa1, xb1, xa2, xb2], 1.0)
        add_term([xa1, xb1], -2.0 * t)

# --- 2) One-hot penalty terms for each face of die A and die B ---
# (sum_v x_v - 1)^2 = -sum_v x_v + 2*sum_{v<w} x_v*x_w + 1   (using x_v^2 = x_v)
for i in range(n_faces):
    vars_i = [xA(i, v) for v in valuesA]
    for v in vars_i:
        add_term([v], -1.0 * penalty)
    for va, vb in itertools.combinations(vars_i, 2):
        add_term([va, vb], 2.0 * penalty)

for j in range(n_faces):
    vars_j = [xB(j, w) for w in valuesB]
    for w in vars_j:
        add_term([w], -1.0 * penalty)
    for wa, wb in itertools.combinations(vars_j, 2):
        add_term([wa, wb], 2.0 * penalty)

# drop exact-zero coefficients (harmless, just keeps the dict smaller)
hubo = {k: v for k, v in hubo.items() if v != 0}

# ----------------------------------------------------------------------
# Solve with OpenJij's HUBO simulated annealing sampler
# ----------------------------------------------------------------------
sampler = oj.SASampler()
response = sampler.sample_hubo(
    hubo,
    vartype="BINARY",
    num_reads=num_reads,
    num_sweeps=num_sweeps,
)

best = response.first.sample
best_energy = response.first.energy

# theoretical energy of a perfect solution, for sanity-checking convergence
# (see derivation: dropped constants = sum(target^2) + penalty * 12)
dropped_constants = sum(t * t for t in target.values()) + penalty * 2 * n_faces
perfect_energy = -dropped_constants
print(f"Theoretical perfect-match energy: {perfect_energy}")
print(f"Best energy found:                {best_energy}")

# ----------------------------------------------------------------------
# Decode & validate
# ----------------------------------------------------------------------
def decode(prefix, values):
    faces = []
    valid = True
    for i in range(n_faces):
        chosen = [v for v in values if best.get(f"{prefix}_{i}_{v}", 0) == 1]
        if len(chosen) != 1:
            valid = False
            faces.append(chosen if chosen else None)
        else:
            faces.append(chosen[0])
    return faces, valid

dieA, validA = decode("A", valuesA)
dieB, validB = decode("B", valuesB)

print(f"Die A faces: {dieA}   (valid one-hot: {validA})")
print(f"Die B faces: {dieB}   (valid one-hot: {validB})")

if validA and validB:
    # verify the sum distribution actually matches
    counts = {}
    for a in dieA:
        for b in dieB:
            counts[a + b] = counts.get(a + b, 0) + 1
    print("Resulting sum distribution:", dict(sorted(counts.items())))
    print("Target sum distribution:   ", dict(sorted(target.items())))
    match = (counts == target)
    print("Match:", match)

    if not match:
        print("-> Did not converge to the global optimum. Try increasing "
              "num_reads / num_sweeps, or raising `penalty` further.")
    elif sorted(dieA) == [1, 2, 3, 4, 5, 6] and sorted(dieB) == [1, 2, 3, 4, 5, 6]:
        print("-> This is the trivial standard-dice solution (shouldn't be "
              "reachable now that VmaxA=4 excludes value 6 -- check VmaxA).")
    else:
        print("-> Non-trivial (Sicherman) solution found and VERIFIED!")
else:
    print("One-hot constraints violated - increase `penalty` or `num_sweeps` and retry.")