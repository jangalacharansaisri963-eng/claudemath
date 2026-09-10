"""Combinatorics module for claudemath.

Pure-Python implementation of counting algorithms, permutations, combinations,
factorials, derangements, integer partitions, set partitions, combinatorial sequences,
Gray codes, and lexicographic ranking/unranking.
"""

from typing import List, Tuple, Optional, Generator, Dict, Any
import math


# ----------------------------------------------------
# 1. Factorials and Extended Products
# ----------------------------------------------------

def factorial(n: int) -> int:
    """Exact factorial n!."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return math.factorial(n)


def double_factorial(n: int) -> int:
    """Double factorial n!! (product of integers of same parity down to 1 or 2)."""
    if n < -1:
        raise ValueError("n must be >= -1")
    if n in (-1, 0, 1):
        return 1
    res = 1
    for k in range(n, 0, -2):
        res *= k
    return res


def multifactorial(n: int, k: int) -> int:
    """Multifactorial n!^(k) stepping by k down to > 0."""
    if k <= 0:
        raise ValueError("step k must be positive")
    if n <= 0:
        return 1
    res = 1
    for val in range(n, 0, -k):
        res *= val
    return res


def rising_factorial(x: float, n: int) -> float:
    """Pochhammer symbol / rising factorial x^{(n)} = x*(x+1)*...*(x+n-1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    res = 1.0
    for i in range(n):
        res *= (x + i)
    return res


def falling_factorial(x: float, n: int) -> float:
    """Falling factorial (x)_n = x*(x-1)*...*(x-n+1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    res = 1.0
    for i in range(n):
        res *= (x - i)
    return res


def subfactorial(n: int) -> int:
    """Number of derangements !n = round(n! / e)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    if n == 1:
        return 0
    d0, d1 = 1, 0
    for i in range(2, n + 1):
        d_next = (i - 1) * (d1 + d0)
        d0, d1 = d1, d_next
    return d1


def primorial(n: int) -> int:
    """Product of all prime numbers <= n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    # Quick sieve
    if n < 2:
        return 1
    primes = [p for p in range(2, n + 1) if all(p % d != 0 for d in range(2, int(math.isqrt(p)) + 1))]
    res = 1
    for p in primes:
        res *= p
    return res


# ----------------------------------------------------
# 2. Permutations and Combinations Counts
# ----------------------------------------------------

def nPr(n: int, r: int) -> int:
    """Number of permutations P(n, r) = n! / (n - r)!."""
    if r < 0 or r > n or n < 0:
        return 0
    return math.perm(n, r)


def nCr(n: int, r: int) -> int:
    """Binomial coefficient C(n, r) = n! / (r! * (n - r)!)."""
    if r < 0 or r > n or n < 0:
        return 0
    return math.comb(n, r)


def nCr_with_repetition(n: int, r: int) -> int:
    """Combinations with replacement C(n + r - 1, r)."""
    if n <= 0 or r < 0:
        return 0
    return math.comb(n + r - 1, r)


def central_binomial_coefficient(n: int) -> int:
    """Central binomial coefficient C(2n, n)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return math.comb(2 * n, n)


def multinomial_count(group_sizes: List[int]) -> int:
    """Number of ways to partition sum(sizes) into distinct groups."""
    total = sum(group_sizes)
    res = 1
    running = 0
    for s in group_sizes:
        running += s
        res *= math.comb(running, s)
    return res


# ----------------------------------------------------
# 3. Special Combinatorial Numbers
# ----------------------------------------------------

def catalan_number(n: int) -> int:
    """Catalan number C_n = C(2n, n) / (n + 1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return math.comb(2 * n, n) // (n + 1)


def motzkin_number(n: int) -> int:
    """Motzkin number M_n: paths from (0,0) to (n,0) with steps (1,1), (1,-1), (1,0)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    m = [0] * (n + 1)
    m[0] = 1
    m[1] = 1
    for i in range(2, n + 1):
        term1 = (2 * i + 1) * m[i - 1]
        term2 = (3 * i - 3) * m[i - 2]
        m[i] = (term1 + term2) // (i + 2)
    return m[n]


def narayana_number(n: int, k: int) -> int:
    """Narayana number N(n, k) = (1 / n) * C(n, k) * C(n, k - 1)."""
    if k < 1 or k > n or n < 1:
        return 0
    return (math.comb(n, k) * math.comb(n, k - 1)) // n


def delannoy_number(m: int, n: int) -> int:
    """Delannoy number D(m, n): grid paths with steps E, N, NE."""
    if m < 0 or n < 0:
        raise ValueError("m and n must be non-negative")
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for j in range(n + 1):
        dp[0][j] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1] + dp[i - 1][j - 1]
    return dp[m][n]


def eulerian_number(n: int, k: int) -> int:
    """Eulerian number A(n, k): permutations of {1..n} with k ascents."""
    if k < 0 or k >= n:
        return 0
    if n == 0:
        return 1 if k == 0 else 0
    total = 0
    for j in range(k + 2):
        term = ((-1) ** j) * math.comb(n + 1, j) * ((k + 1 - j) ** n)
        total += term
    return total


def lah_number(n: int, k: int) -> int:
    """Unsigned Lah number L(n, k) = (n! / k!) * C(n-1, k-1)."""
    if n == 0 and k == 0:
        return 1
    if k < 1 or k > n:
        return 0
    return (math.factorial(n) // math.factorial(k)) * math.comb(n - 1, k - 1)


def fibonacci_number(n: int) -> int:
    """Exact n-th Fibonacci number F_n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def jacobsthal_number(n: int) -> int:
    """Jacobsthal number J_n = (2^n - (-1)^n) / 3."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return ((1 << n) - ((-1) ** n)) // 3


def super_catalan_number(n: int) -> int:
    """Super-Catalan (Schroeder-Hipparchus) number S_n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    s = [1, 1]
    for i in range(2, n + 1):
        val = ((6 * i - 3) * s[-1] - (i - 2) * s[-2]) // (i + 1)
        s.append(val)
    return s[n]


# ----------------------------------------------------
# 4. Permutations and Combinations Generation
# ----------------------------------------------------

def next_lexicographical_permutation(arr: List[int]) -> bool:
    """In-place step to next lexicographical permutation. Returns False if already highest."""
    n = len(arr)
    # 1. Find largest index i such that arr[i] < arr[i + 1]
    i = n - 2
    while i >= 0 and arr[i] >= arr[i + 1]:
        i -= 1
    if i < 0:
        arr.reverse()
        return False
    # 2. Find largest index j > i such that arr[i] < arr[j]
    j = n - 1
    while arr[j] <= arr[i]:
        j -= 1
    arr[i], arr[j] = arr[j], arr[i]
    # 3. Reverse from i + 1 to end
    arr[i + 1:] = reversed(arr[i + 1:])
    return True


def prev_lexicographical_permutation(arr: List[int]) -> bool:
    """In-place step to previous lexicographical permutation."""
    n = len(arr)
    i = n - 2
    while i >= 0 and arr[i] <= arr[i + 1]:
        i -= 1
    if i < 0:
        arr.reverse()
        return False
    j = n - 1
    while arr[j] >= arr[i]:
        j -= 1
    arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1:] = reversed(arr[i + 1:])
    return True


def permutation_to_lehmer_code(p: List[int]) -> List[int]:
    """Compute Lehmer code (factoradic representation) of permutation."""
    n = len(p)
    code = [0] * n
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            if p[j] < p[i]:
                count += 1
        code[i] = count
    return code


def lehmer_code_to_permutation(code: List[int]) -> List[int]:
    """Reconstruct permutation from its Lehmer code."""
    n = len(code)
    available = list(range(n))
    perm = []
    for c in code:
        perm.append(available.pop(c))
    return perm


def permutation_rank(p: List[int]) -> int:
    """0-based lexicographical rank of permutation p of {0, 1, ..., n-1}."""
    code = permutation_to_lehmer_code(p)
    n = len(p)
    rank = 0
    fact = 1
    for i in range(n):
        rank += code[n - 1 - i] * fact
        fact *= (i + 1)
    return rank


def permutation_unrank(n: int, rank: int) -> List[int]:
    """Reconstruct permutation of {0, 1, ..., n-1} with given 0-based rank."""
    code = [0] * n
    for i in range(1, n + 1):
        code[n - i] = rank % i
        rank //= i
    return lehmer_code_to_permutation(code)


def combination_rank(comb: List[int], n: int) -> int:
    """Rank k-combination (sorted 0-based subset of {0, ..., n-1}) using combinatorial number system."""
    k = len(comb)
    rank = 0
    for i, c in enumerate(comb):
        rank += math.comb(c, i + 1)
    return rank


def combination_unrank(rank: int, n: int, k: int) -> List[int]:
    """Unrank combination of k elements from {0, ..., n-1}."""
    comb = []
    curr = n - 1
    rem = rank
    for i in range(k, 0, -1):
        while math.comb(curr, i) > rem:
            curr -= 1
        comb.append(curr)
        rem -= math.comb(curr, i)
        curr -= 1
    return sorted(comb)


def count_inversions(arr: List[int]) -> int:
    """Count number of inverted pairs (i < j with arr[i] > arr[j])."""
    inv = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def kendall_tau_distance(p1: List[int], p2: List[int]) -> int:
    """Number of discordant pairs between two rankings."""
    pos = {val: i for i, val in enumerate(p2)}
    mapped = [pos[val] for val in p1]
    return count_inversions(mapped)


# ----------------------------------------------------
# 5. Partitions, Compositions, and Set Partitions
# ----------------------------------------------------

def integer_partitions_generate(n: int) -> List[List[int]]:
    """Generate all partitions of positive integer n in non-increasing order."""
    if n <= 0:
        return [[]] if n == 0 else []
    res = []
    def helper(remaining: int, max_val: int, current: List[int]):
        if remaining == 0:
            res.append(list(current))
            return
        for val in range(min(remaining, max_val), 0, -1):
            current.append(val)
            helper(remaining - val, val, current)
            current.pop()
    helper(n, n, [])
    return res


def conjugate_partition(partition: List[int]) -> List[int]:
    """Conjugate (transposed Ferrers diagram) of integer partition."""
    if not partition:
        return []
    max_val = partition[0]
    conj = []
    for i in range(1, max_val + 1):
        count = sum(1 for part in partition if part >= i)
        conj.append(count)
    return conj


def compositions_count(n: int, k: int) -> int:
    """Number of compositions of n into k strictly positive parts: C(n - 1, k - 1)."""
    if n < k or k < 1:
        return 0
    return math.comb(n - 1, k - 1)


def weak_compositions_count(n: int, k: int) -> int:
    """Number of weak compositions of n into k non-negative parts: C(n + k - 1, k - 1)."""
    if n < 0 or k < 1:
        return 0
    return math.comb(n + k - 1, k - 1)


def set_partitions_generate(elements: List[Any]) -> List[List[List[Any]]]:
    """Generate all set partitions of a list of elements."""
    if not elements:
        return [[]]
    first = elements[0]
    rest = elements[1:]
    rest_partitions = set_partitions_generate(rest)
    result = []
    for partition in rest_partitions:
        # Option 1: add first as a singleton block
        result.append([[first]] + partition)
        # Option 2: insert first into one of the existing blocks
        for i in range(len(partition)):
            new_partition = [block[:] for block in partition]
            new_partition[i].append(first)
            result.append(new_partition)
    return result


# ----------------------------------------------------
# 6. Gray Codes and Bit Combinatorics
# ----------------------------------------------------

def binary_reflected_gray_code(n: int) -> List[int]:
    """Generate n-bit Binary Reflected Gray Code sequence as integers."""
    if n <= 0:
        return [0]
    return [i ^ (i >> 1) for i in range(1 << n)]


def binary_to_gray(n: int) -> int:
    """Convert binary integer to Gray code: n ^ (n >> 1)."""
    return n ^ (n >> 1)


def gray_to_binary(g: int) -> int:
    """Convert Gray code to standard binary integer."""
    mask = g >> 1
    while mask != 0:
        g ^= mask
        mask >>= 1
    return g


def popcount(n: int) -> int:
    """Count number of set bits (Hamming weight)."""
    return bin(n).count('1')


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers (number of differing bits)."""
    return popcount(a ^ b)
def generate_combinations(arr: List[Any], k: int) -> List[List[Any]]:
    """Generate all k-element combinations from arr without replacement."""
    n = len(arr)
    if k < 0 or k > n:
        return []
    if k == 0:
        return [[]]
    res = []
    def backtrack(start: int, curr: List[Any]):
        if len(curr) == k:
            res.append(list(curr))
            return
        for i in range(start, n):
            curr.append(arr[i])
            backtrack(i + 1, curr)
            curr.pop()
    backtrack(0, [])
    return res


def generate_permutations(arr: List[Any]) -> List[List[Any]]:
    """Generate all n! permutations of distinct elements in arr."""
    n = len(arr)
    res = []
    used = [False] * n
    def backtrack(curr: List[Any]):
        if len(curr) == n:
            res.append(list(curr))
            return
        for i in range(n):
            if not used[i]:
                used[i] = True
                curr.append(arr[i])
                backtrack(curr)
                curr.pop()
                used[i] = False
    backtrack([])
    return res


def generate_cartesian_product(pools: List[List[Any]]) -> List[List[Any]]:
    """Compute Cartesian product of multiple lists."""
    if not pools:
        return [[]]
    res = [[]]
    for pool in pools:
        res = [x + [y] for x in res for y in pool]
    return res


def generate_powerset(arr: List[Any]) -> List[List[Any]]:
    """Generate powerset (all 2^n subsets) of arr."""
    n = len(arr)
    res = []
    for i in range(1 << n):
        subset = [arr[j] for j in range(n) if (i & (1 << j))]
        res.append(subset)
    return res


def generate_derangements(arr: List[int]) -> List[List[int]]:
    """Generate all derangements of distinct elements in arr."""
    n = len(arr)
    res = []
    used = [False] * n
    def backtrack(idx: int, curr: List[int]):
        if idx == n:
            res.append(list(curr))
            return
        for i in range(n):
            if not used[i] and arr[i] != arr[idx]:
                used[i] = True
                curr.append(arr[i])
                backtrack(idx + 1, curr)
                curr.pop()
                used[i] = False
    backtrack(0, [])
    return res


def lucas_binomial_mod_p(n: int, k: int, p: int) -> int:
    """Compute C(n, k) mod p (for prime p) using Lucas' theorem."""
    if k < 0 or k > n:
        return 0
    res = 1
    while n > 0 or k > 0:
        n_i = n % p
        k_i = k % p
        if k_i > n_i:
            return 0
        res = (res * math.comb(n_i, k_i)) % p
        n //= p
        k //= p
    return res


def kummers_theorem_carry_count(n: int, k: int, p: int) -> int:
    """Count power of prime p dividing C(n, k) using Kummer's carries in base p."""
    carries = 0
    carry = 0
    m = n - k
    while k > 0 or m > 0 or carry > 0:
        s = (k % p) + (m % p) + carry
        carry = s // p
        if carry > 0:
            carries += 1
        k //= p
        m //= p
    return carries


def vandermonde_identity_sum(m: int, n: int, r: int) -> int:
    """Vandermonde identity: sum_{k=0}^r C(m, k) * C(n, r - k) = C(m + n, r)."""
    return math.comb(m + n, r)


def bell_number_exact(n: int) -> int:
    """Exact n-th Bell number B_n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    row = [1]
    for i in range(1, n + 1):
        next_row = [row[-1]]
        for j in range(len(row)):
            next_row.append(next_row[-1] + row[j])
        row = next_row
    return row[0]


def stirling_first_kind_exact(n: int, k: int) -> int:
    """Unsigned Stirling numbers of first kind |c(n, k)|."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i - 1][j - 1] + (i - 1) * dp[i - 1][j]
    return dp[n][k]


def stirling_second_kind_exact(n: int, k: int) -> int:
    """Stirling numbers of second kind S(n, k)."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    total = 0
    for j in range(k + 1):
        term = ((-1) ** (k - j)) * math.comb(k, j) * (j ** n)
        total += term
    return total // math.factorial(k)


def fubini_number(n: int) -> int:
    """Compute n-th Fubini number (ordered Bell number) sum_{k=0}^n k! * S(n, k)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    return sum(math.factorial(k) * stirling_second_kind_exact(n, k) for k in range(1, n + 1))


def young_tableau_hook_length_formula(partition: List[int]) -> int:
    """Number of standard Young tableaux of given integer partition shape using Hook Length Formula."""
    n = sum(partition)
    num_rows = len(partition)
    hook_product = 1
    for i in range(num_rows):
        for j in range(partition[i]):
            # Leg length: number of cells below (i, j)
            leg = sum(1 for r in range(i + 1, num_rows) if partition[r] > j)
            # Arm length: number of cells to right of (i, j)
            arm = partition[i] - 1 - j
            hook = leg + arm + 1
            hook_product *= hook
    return math.factorial(n) // hook_product


def dyck_paths_count(n: int) -> int:
    """Number of Dyck paths of length 2n equals Catalan number C_n."""
    return catalan_number(n)


def dyck_path_is_valid(path: str) -> bool:
    """Check if string of 'U' and 'D' is a valid Dyck path."""
    height = 0
    for step in path:
        if step == 'U':
            height += 1
        elif step == 'D':
            height -= 1
            if height < 0:
                return False
        else:
            return False
    return height == 0


def gosper_next_bit_permutation(v: int) -> int:
    """Compute next higher integer with same number of set bits (Gosper's hack)."""
    t = v | (v - 1)
    return (t + 1) | (((~t & -(~t)) - 1) >> (popcount(v ^ (v - 1)) + 1))


def prufer_sequence_to_tree_edges(seq: List[int]) -> List[Tuple[int, int]]:
    """Convert Prufer sequence of length n-2 to tree edge list (nodes 0..n-1)."""
    n = len(seq) + 2
    degree = [1] * n
    for node in seq:
        degree[node] += 1
    edges = []
    for node in seq:
        for i in range(n):
            if degree[i] == 1:
                edges.append((node, i))
                degree[node] -= 1
                degree[i] -= 1
                break
    u, v = [i for i in range(n) if degree[i] == 1]
    edges.append((u, v))
    return edges


def tree_edges_to_prufer_sequence(edges: List[Tuple[int, int]], n: int) -> List[int]:
    """Convert tree edges on n vertices (0..n-1) to Prufer sequence of length n-2."""
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    prufer = []
    for _ in range(n - 2):
        # Find leaf with smallest label
        leaf = min(i for i in range(n) if len(adj[i]) == 1)
        neighbor = list(adj[leaf])[0]
        prufer.append(neighbor)
        adj[leaf].remove(neighbor)
        adj[neighbor].remove(leaf)
    return prufer


def composition_generate(n: int, k: int) -> List[List[int]]:
    """Generate all compositions of n into k strictly positive integers."""
    if n < k or k < 1:
        return []
    if k == 1:
        return [[n]]
    res = []
    def helper(rem: int, parts_left: int, current: List[int]):
        if parts_left == 1:
            res.append(current + [rem])
            return
        for val in range(1, rem - parts_left + 2):
            helper(rem - val, parts_left - 1, current + [val])
    helper(n, k, [])
    return res


def macmahon_box_plane_partitions(r: int, s: int, t: int) -> int:
    """MacMahon's box formula for number of plane partitions fitting in r x s x t box."""
    num = 1
    for i in range(1, r + 1):
        for j in range(1, s + 1):
            for k in range(1, t + 1):
                num *= (i + j + k - 1)
    den = 1
    for i in range(1, r + 1):
        for j in range(1, s + 1):
            for k in range(1, t + 1):
                den *= (i + j + k - 2) if (i + j + k - 2) > 0 else 1
    # Standard formula: prod_{i=1}^r prod_{j=1}^s (i + j + t - 1) / (i + j - 1)
    prod = 1
    for i in range(1, r + 1):
        for j in range(1, s + 1):
            prod = (prod * (i + j + t - 1)) // (i + j - 1)
    return prod
def multiset_permutations_count(counts: List[int]) -> int:
    """Number of distinct permutations of multiset with item counts: (sum(c_i))! / prod(c_i!)."""
    total = sum(counts)
    res = 1
    running = 0
    for c in counts:
        running += c
        res *= math.comb(running, c)
    return res


def stars_and_bars_count(n: int, k: int) -> int:
    """Ways to distribute n indistinguishable items into k distinguishable bins: C(n + k - 1, k - 1)."""
    if n < 0 or k < 1:
        return 0
    return math.comb(n + k - 1, k - 1)


def stars_and_bars_non_empty(n: int, k: int) -> int:
    """Ways to distribute n indistinguishable items into k non-empty distinguishable bins: C(n - 1, k - 1)."""
    if n < k or k < 1:
        return 0
    return math.comb(n - 1, k - 1)


def telephone_number(n: int) -> int:
    """Number of involutions (permutations that are their own inverse) in S_n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    a, b = 1, 1
    for i in range(2, n + 1):
        a, b = b, b + (i - 1) * a
    return b


def necklace_count_fixed_k(n: int, k: int) -> int:
    """Number of distinct necklaces of length n using k colors under rotational symmetry (Burnside)."""
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    # sum_{d|n} phi(d) * k^(n/d) / n
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            phi_d = sum(1 for x in range(1, d + 1) if math.gcd(x, d) == 1)
            total += phi_d * (k ** (n // d))
    return total // n


def bracelet_count_fixed_k(n: int, k: int) -> int:
    """Number of distinct bracelets of length n with k colors under rotation and reflection."""
    necklaces = necklace_count_fixed_k(n, k)
    if n % 2 == 1:
        reflections = k ** ((n + 1) // 2)
    else:
        reflections = (k ** (n // 2) + k ** (n // 2 + 1)) // 2
    return (necklaces + reflections) // 2


def schroeder_little_number(n: int) -> int:
    """Little Schroeder number s_n: s_0 = 1, s_n = S_n / 2 for n >= 1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    return super_catalan_number(n) // 2


def derangement_probability(n: int) -> float:
    """Probability that a uniform random permutation in S_n is a derangement (!n / n!)."""
    if n <= 0:
        return 1.0
    return subfactorial(n) / math.factorial(n)


def subset_lexicographic_rank(subset: List[int], n: int) -> int:
    """0-based rank of subset of {0, ..., n-1} in lexicographical subset order."""
    rank = 0
    for elem in subset:
        if 0 <= elem < n:
            rank |= (1 << elem)
    return rank


def subset_lexicographic_unrank(rank: int, n: int) -> List[int]:
    """Decode integer bitmask into subset of {0, ..., n-1}."""
    return [i for i in range(n) if (rank & (1 << i))]


def eulerian_polynomial_eval(n: int, t: float) -> float:
    """Evaluate Eulerian polynomial A_n(t) = sum_{k=0}^{n-1} A(n, k) * t^k."""
    if n == 0:
        return 1.0
    return sum(eulerian_number(n, k) * (t ** k) for k in range(n))


def alternating_permutation_count_andre(n: int) -> int:
    """Andre's theorem: count alternating (zigzag) permutations of length n (Euler/secant/tangent numbers)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    # Seidel-Entringer triangle
    row = [1]
    for k in range(1, n + 1):
        new_row = [0] * (k + 1)
        for i in range(1, k + 1):
            new_row[i] = new_row[i - 1] + row[k - i]
        row = new_row
    return row[-1]
