"""Discrete mathematics module for claudemath.

Pure-Python implementation of set theory, propositional logic and truth tables,
DPLL SAT solver, binary relations, order theory (posets, lattices, Hasse diagrams),
and recurrence relations (Master theorem, linear constant-coefficient solvers).
"""

from typing import List, Tuple, Set, Dict, Any, Optional, Callable
import math


# ----------------------------------------------------
# 1. Set Theory Operations
# ----------------------------------------------------

def set_union(s1: Set[Any], s2: Set[Any]) -> Set[Any]:
    """Union of two sets s1 U s2."""
    return s1 | s2


def set_intersection(s1: Set[Any], s2: Set[Any]) -> Set[Any]:
    """Intersection of two sets s1 (cap) s2."""
    return s1 & s2


def set_difference(s1: Set[Any], s2: Set[Any]) -> Set[Any]:
    """Set difference s1 \ s2."""
    return s1 - s2


def set_symmetric_difference(s1: Set[Any], s2: Set[Any]) -> Set[Any]:
    """Symmetric difference s1 (Delta) s2."""
    return s1 ^ s2


def set_cartesian_product(s1: Set[Any], s2: Set[Any]) -> Set[Tuple[Any, Any]]:
    """Cartesian product s1 x s2."""
    return {(a, b) for a in s1 for b in s2}


def set_power_set(s: Set[Any]) -> List[Set[Any]]:
    """Power set P(s) containing all 2^|s| subsets."""
    elem_list = list(s)
    n = len(elem_list)
    res = []
    for i in range(1 << n):
        subset = {elem_list[j] for j in range(n) if (i & (1 << j))}
        res.append(subset)
    return res


def is_subset(sub: Set[Any], parent: Set[Any]) -> bool:
    """Check if sub is a subset of parent."""
    return sub.issubset(parent)


def is_proper_subset(sub: Set[Any], parent: Set[Any]) -> bool:
    """Check if sub is a strict proper subset of parent."""
    return sub < parent


def is_disjoint(s1: Set[Any], s2: Set[Any]) -> bool:
    """Check if s1 and s2 share no common elements."""
    return s1.isdisjoint(s2)


def jaccard_similarity(s1: Set[Any], s2: Set[Any]) -> float:
    """Jaccard index J(A, B) = |A cap B| / |A cup B|."""
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)


def jaccard_distance(s1: Set[Any], s2: Set[Any]) -> float:
    """Jaccard distance = 1 - J(A, B)."""
    return 1.0 - jaccard_similarity(s1, s2)


def dice_similarity(s1: Set[Any], s2: Set[Any]) -> float:
    """Sorensen-Dice coefficient: 2 * |A cap B| / (|A| + |B|)."""
    total = len(s1) + len(s2)
    if total == 0:
        return 1.0
    return (2.0 * len(s1 & s2)) / total


# ----------------------------------------------------
# 2. Propositional Logic and Boolean Algebra
# ----------------------------------------------------

def logic_and(p: bool, q: bool) -> bool:
    """Logical conjunction p AND q."""
    return p and q


def logic_or(p: bool, q: bool) -> bool:
    """Logical disjunction p OR q."""
    return p or q


def logic_not(p: bool) -> bool:
    """Logical negation NOT p."""
    return not p


def logic_implies(p: bool, q: bool) -> bool:
    """Material conditional p -> q (not p or q)."""
    return (not p) or q


def logic_iff(p: bool, q: bool) -> bool:
    """Logical biconditional p <-> q."""
    return p == q


def logic_xor(p: bool, q: bool) -> bool:
    """Exclusive OR p XOR q."""
    return p != q


def logic_nand(p: bool, q: bool) -> bool:
    """Sheffer stroke NAND: not (p and q)."""
    return not (p and q)


def logic_nor(p: bool, q: bool) -> bool:
    """Peirce arrow NOR: not (p or q)."""
    return not (p or q)


def generate_truth_table(variables: List[str], formula: Callable[[Dict[str, bool]], bool]) -> List[Tuple[Dict[str, bool], bool]]:
    """Generate truth table for boolean formula over variables."""
    n = len(variables)
    table = []
    for i in range(1 << n):
        assignment = {variables[j]: bool(i & (1 << (n - 1 - j))) for j in range(n)}
        result = formula(assignment)
        table.append((assignment, result))
    return table


def is_tautology(variables: List[str], formula: Callable[[Dict[str, bool]], bool]) -> bool:
    """Check if boolean formula evaluates to True for all truth assignments."""
    table = generate_truth_table(variables, formula)
    return all(res for _, res in table)


def is_contradiction(variables: List[str], formula: Callable[[Dict[str, bool]], bool]) -> bool:
    """Check if boolean formula evaluates to False for all assignments."""
    table = generate_truth_table(variables, formula)
    return all(not res for _, res in table)


def is_satisfiable(variables: List[str], formula: Callable[[Dict[str, bool]], bool]) -> bool:
    """Check if boolean formula evaluates to True for at least one assignment."""
    table = generate_truth_table(variables, formula)
    return any(res for _, res in table)


def evaluate_cnf(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
    """Evaluate Conjunctive Normal Form (CNF) where positive int is var, negative is neg var."""
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            val = assignment.get(var, False)
            if lit < 0:
                val = not val
            if val:
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False
    return True


def dpll_sat_solver(clauses: List[List[int]], assignment: Optional[Dict[int, bool]] = None) -> Optional[Dict[int, bool]]:
    """Davis-Putnam-Logemann-Loveland (DPLL) pure-Python SAT solver."""
    if assignment is None:
        assignment = {}

    # Simplify clauses under current assignment
    new_clauses = []
    for c in clauses:
        c_sat = False
        new_c = []
        for lit in c:
            var = abs(lit)
            if var in assignment:
                val = assignment[var] if lit > 0 else not assignment[var]
                if val:
                    c_sat = True
                    break
            else:
                new_c.append(lit)
        if not c_sat:
            if not new_c:
                return None  # Empty clause: conflict
            new_clauses.append(new_c)

    if not new_clauses:
        return assignment

    # Unit propagation
    for c in new_clauses:
        if len(c) == 1:
            lit = c[0]
            var = abs(lit)
            val = (lit > 0)
            next_assign = dict(assignment)
            next_assign[var] = val
            return dpll_sat_solver(new_clauses, next_assign)

    # Pure literal elimination
    all_lits = [lit for c in new_clauses for lit in c]
    all_vars = {abs(lit) for lit in all_lits}
    for var in all_vars:
        has_pos = var in all_lits
        has_neg = -var in all_lits
        if has_pos and not has_neg:
            next_assign = dict(assignment)
            next_assign[var] = True
            return dpll_sat_solver(new_clauses, next_assign)
        if has_neg and not has_pos:
            next_assign = dict(assignment)
            next_assign[var] = False
            return dpll_sat_solver(new_clauses, next_assign)

    # Choose branching variable
    branch_var = abs(new_clauses[0][0])
    assign_true = dict(assignment)
    assign_true[branch_var] = True
    sol = dpll_sat_solver(new_clauses, assign_true)
    if sol is not None:
        return sol
    assign_false = dict(assignment)
    assign_false[branch_var] = False
    return dpll_sat_solver(new_clauses, assign_false)


# ----------------------------------------------------
# 3. Binary Relations and Properties
# ----------------------------------------------------

def is_reflexive_relation(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> bool:
    """Check if relation R on universe is reflexive (a R a for all a in universe)."""
    return all((a, a) in relation for a in universe)


def is_irreflexive_relation(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> bool:
    """Check if relation R is irreflexive ((a, a) not in R for all a)."""
    return all((a, a) not in relation for a in universe)


def is_symmetric_relation(relation: Set[Tuple[Any, Any]]) -> bool:
    """Check if relation is symmetric (a R b implies b R a)."""
    return all((b, a) in relation for a, b in relation)


def is_antisymmetric_relation(relation: Set[Tuple[Any, Any]]) -> bool:
    """Check if relation is antisymmetric (a R b and b R a implies a == b)."""
    return all(a == b for a, b in relation if (b, a) in relation)


def is_asymmetric_relation(relation: Set[Tuple[Any, Any]]) -> bool:
    """Check if relation is asymmetric (a R b implies (b, a) not in R)."""
    return all((b, a) not in relation for a, b in relation)


def is_transitive_relation(relation: Set[Tuple[Any, Any]]) -> bool:
    """Check if relation is transitive (a R b and b R c implies a R c)."""
    for a, b in relation:
        for c, d in relation:
            if b == c:
                if (a, d) not in relation:
                    return False
    return True


def is_equivalence_relation(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> bool:
    """Check if relation is reflexive, symmetric, and transitive."""
    return (is_reflexive_relation(relation, universe) and
            is_symmetric_relation(relation) and
            is_transitive_relation(relation))


def equivalence_classes(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> List[Set[Any]]:
    """Compute partition of universe into equivalence classes under R."""
    if not is_equivalence_relation(relation, universe):
        raise ValueError("Relation is not an equivalence relation")
    classes = []
    visited = set()
    for a in universe:
        if a not in visited:
            eq_class = {b for b in universe if (a, b) in relation}
            classes.append(eq_class)
            visited.update(eq_class)
    return classes


def reflexive_closure(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Set[Tuple[Any, Any]]:
    """Reflexive closure of relation R: R cup {(a, a) | a in universe}."""
    return relation | {(a, a) for a in universe}


def symmetric_closure(relation: Set[Tuple[Any, Any]]) -> Set[Tuple[Any, Any]]:
    """Symmetric closure of relation R: R cup R^{-1}."""
    return relation | {(b, a) for a, b in relation}


def transitive_closure_warshall(relation: Set[Tuple[Any, Any]], universe: List[Any]) -> Set[Tuple[Any, Any]]:
    """Transitive closure using Warshall's algorithm."""
    n = len(universe)
    pos = {elem: i for i, elem in enumerate(universe)}
    reach = [[False] * n for _ in range(n)]
    for a, b in relation:
        if a in pos and b in pos:
            reach[pos[a]][pos[b]] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    res = set()
    for i in range(n):
        for j in range(n):
            if reach[i][j]:
                res.add((universe[i], universe[j]))
    return res


# ----------------------------------------------------
# 4. Posets, Lattices, and Order Theory
# ----------------------------------------------------

def is_partial_order(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> bool:
    """Check if relation is a partial order (reflexive, antisymmetric, transitive)."""
    return (is_reflexive_relation(relation, universe) and
            is_antisymmetric_relation(relation) and
            is_transitive_relation(relation))


def is_strict_partial_order(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> bool:
    """Check if relation is strict partial order (irreflexive, transitive)."""
    return is_irreflexive_relation(relation, universe) and is_transitive_relation(relation)


def is_total_order(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> bool:
    """Check if relation is a total (linear) order."""
    if not is_partial_order(relation, universe):
        return False
    for a in universe:
        for b in universe:
            if (a, b) not in relation and (b, a) not in relation:
                return False
    return True


def hasse_diagram_covering_relations(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Set[Tuple[Any, Any]]:
    """Compute covering relation (edges in Hasse diagram of poset)."""
    covers = set()
    for a, b in relation:
        if a != b:
            # Check if there is no c such that a < c < b
            intermediate = False
            for c in universe:
                if c != a and c != b:
                    if (a, c) in relation and (c, b) in relation:
                        intermediate = True
                        break
            if not intermediate:
                covers.add((a, b))
    return covers


def topological_sort_dag(nodes: List[Any], edges: List[Tuple[Any, Any]]) -> List[Any]:
    """Kahn's algorithm for topological sorting of directed acyclic graph."""
    in_degree = {node: 0 for node in nodes}
    adj = {node: [] for node in nodes}
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    queue = [node for node in nodes if in_degree[node] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(order) != len(nodes):
        raise ValueError("Graph contains cycles; no topological ordering exists")
    return order


# ----------------------------------------------------
# 5. Recurrence Relations and Asymptotics
# ----------------------------------------------------

def master_theorem_complexity(a: float, b: float, d: float) -> str:
    """Analyze T(n) = a * T(n/b) + Theta(n^d) via the Master Theorem."""
    if a <= 0 or b <= 1 or d < 0:
        raise ValueError("Invalid parameters: require a >= 1, b > 1, d >= 0")
    crit = math.log(a, b)
    if abs(crit - d) < 1e-9:
        return f"Theta(n^{d} * log(n))"
    elif crit > d:
        return f"Theta(n^{crit:.3f})"
    else:
        return f"Theta(n^{d})"


def solve_second_order_linear_recurrence(c1: float, c2: float, a0: float, a1: float, n: int) -> float:
    """Solve homogeneous recurrence a_n = c1 * a_{n-1} + c2 * a_{n-2}."""
    if n == 0:
        return a0
    if n == 1:
        return a1
    # Characteristic equation r^2 - c1*r - c2 = 0
    disc = c1 * c1 + 4.0 * c2
    if abs(disc) < 1e-12:
        # Repeated root r = c1 / 2
        r = c1 / 2.0
        # a_n = (A + B*n) * r^n
        A = a0
        B = (a1 / r - a0) if r != 0 else 0.0
        return (A + B * n) * (r ** n)
    elif disc > 0:
        r1 = (c1 + math.sqrt(disc)) / 2.0
        r2 = (c1 - math.sqrt(disc)) / 2.0
        # a_n = A * r1^n + B * r2^n
        # A + B = a0
        # A*r1 + B*r2 = a1
        B = (a1 - a0 * r1) / (r2 - r1)
        A = a0 - B
        return A * (r1 ** n) + B * (r2 ** n)
    else:
        # Complex roots r = alpha +/- i*beta = R * e^(+/- i*theta)
        alpha = c1 / 2.0
        beta = math.sqrt(-disc) / 2.0
        R = math.hypot(alpha, beta)
        theta = math.atan2(beta, alpha)
        # a_n = R^n * (A * cos(n*theta) + B * sin(n*theta))
        A = a0
        B = (a1 - a0 * R * math.cos(theta)) / (R * math.sin(theta))
        return (R ** n) * (A * math.cos(n * theta) + B * math.sin(n * theta))
def set_overlap_coefficient(s1: Set[Any], s2: Set[Any]) -> float:
    """Overlap (Szymkiewicz-Simpson) coefficient: |s1 cap s2| / min(|s1|, |s2|)."""
    m = min(len(s1), len(s2))
    if m == 0:
        return 1.0
    return len(s1 & s2) / m


def boolean_majority_function(bits: List[bool]) -> bool:
    """Boolean majority function: True if more than half of bits are True."""
    return sum(bits) > len(bits) // 2


def boolean_parity_function(bits: List[bool]) -> bool:
    """Boolean parity (odd parity) function: True if odd number of bits are True."""
    return sum(bits) % 2 == 1


def resolution_step(clause1: List[int], clause2: List[int]) -> Optional[List[int]]:
    """Perform a single propositional resolution step between two clauses."""
    resolvents = []
    for lit in clause1:
        if -lit in clause2:
            new_clause = set(clause1) - {lit} | (set(clause2) - {-lit})
            # Tautology check
            if any(-x in new_clause for x in new_clause):
                continue
            return sorted(new_clause)
    return None


def poset_maximal_elements(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Set[Any]:
    """Find maximal elements in poset (no element strictly greater)."""
    return {a for a in universe if not any((a, b) in relation and a != b for b in universe)}


def poset_minimal_elements(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Set[Any]:
    """Find minimal elements in poset (no element strictly less)."""
    return {a for a in universe if not any((b, a) in relation and a != b for b in universe)}


def poset_greatest_element(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Optional[Any]:
    """Find greatest element (top / 1) in poset if it exists."""
    for a in universe:
        if all((b, a) in relation for b in universe):
            return a
    return None


def poset_least_element(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Optional[Any]:
    """Find least element (bottom / 0) in poset if it exists."""
    for a in universe:
        if all((a, b) in relation for b in universe):
            return a
    return None


def lattice_meet(a: Any, b: Any, relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Optional[Any]:
    """Greatest lower bound (infimum / meet a ^ b)."""
    lower_bounds = {x for x in universe if (x, a) in relation and (x, b) in relation}
    for lb in lower_bounds:
        if all((x, lb) in relation for x in lower_bounds):
            return lb
    return None


def lattice_join(a: Any, b: Any, relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> Optional[Any]:
    """Least upper bound (supremum / join a v b)."""
    upper_bounds = {x for x in universe if (a, x) in relation and (b, x) in relation}
    for ub in upper_bounds:
        if all((ub, x) in relation for x in upper_bounds):
            return ub
    return None


def poset_is_lattice(relation: Set[Tuple[Any, Any]], universe: Set[Any]) -> bool:
    """Check if poset is a lattice (every pair of elements has unique meet and join)."""
    elem_list = list(universe)
    for i in range(len(elem_list)):
        for j in range(i + 1, len(elem_list)):
            a, b = elem_list[i], elem_list[j]
            if lattice_meet(a, b, relation, universe) is None:
                return False
            if lattice_join(a, b, relation, universe) is None:
                return False
    return True


def linear_recurrence_eval_matrix(coeffs: List[float], initials: List[float], n: int) -> float:
    """Fast evaluation of order-k linear recurrence a_n = sum c_i * a_{n-1-i} using companion matrix power."""
    k = len(coeffs)
    if n < k:
        return initials[n]
    # Matrix of size k x k
    M = [[0.0] * k for _ in range(k)]
    M[0] = list(coeffs)
    for i in range(1, k):
        M[i][i - 1] = 1.0

    # Binary exponentiation M^(n - k + 1)
    power = n - k + 1
    # Identity matrix
    res = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    base = [row[:] for row in M]
    while power > 0:
        if power % 2 == 1:
            # res * base
            new_res = [[sum(res[i][p] * base[p][j] for p in range(k)) for j in range(k)] for i in range(k)]
            res = new_res
        # base * base
        new_base = [[sum(base[i][p] * base[p][j] for p in range(k)) for j in range(k)] for i in range(k)]
        base = new_base
        power //= 2

    # Vector product with reversed initials
    state = list(reversed(initials))
    ans = sum(res[0][j] * state[j] for j in range(k))
    return ans
