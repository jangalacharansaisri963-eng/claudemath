"""Graph theory module for claudemath.

Pure-Python implementation of graph representations, traversals (BFS, DFS),
connectivity (SCC via Tarjan and Kosaraju), shortest paths (Dijkstra, Bellman-Ford,
Floyd-Warshall, A*), minimum spanning trees (Kruskal, Prim), maximum flow (Edmonds-Karp),
maximum bipartite matching (Hopcroft-Karp), centralities (PageRank, Betweenness, Closeness),
and Eulerian/Hamiltonian path algorithms.
"""

from typing import List, Tuple, Dict, Set, Optional, Any, Callable
import math
import collections
import heapq


# ----------------------------------------------------
# 1. Disjoint Set Union (Union-Find)
# ----------------------------------------------------

class DisjointSetUnion:
    """Disjoint Set Union (DSU) with union by rank and path compression."""
    def __init__(self, elements: List[Any]):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x: Any) -> Any:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: Any, y: Any) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


# ----------------------------------------------------
# 2. Graph Representations and Conversions
# ----------------------------------------------------

def adjacency_matrix_to_adjacency_list(adj_matrix: List[List[float]]) -> Dict[int, List[Tuple[int, float]]]:
    """Convert adjacency matrix to weighted adjacency list."""
    n = len(adj_matrix)
    adj_list: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            w = adj_matrix[i][j]
            if w != 0:
                adj_list[i].append((j, w))
    return adj_list


def adjacency_list_to_adjacency_matrix(adj_list: Dict[int, List[Tuple[int, float]]], n: int) -> List[List[float]]:
    """Convert weighted adjacency list to n x n adjacency matrix."""
    matrix = [[0.0] * n for _ in range(n)]
    for u, neighbors in adj_list.items():
        for v, w in neighbors:
            matrix[u][v] = w
    return matrix


def edge_list_to_adjacency_list(edges: List[Tuple[int, int, float]], directed: bool = False) -> Dict[int, List[Tuple[int, float]]]:
    """Convert edge list (u, v, weight) to adjacency list."""
    adj_list: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
    for u, v, w in edges:
        adj_list[u].append((v, w))
        if not directed:
            adj_list[v].append((u, w))
    return dict(adj_list)


# ----------------------------------------------------
# 3. Graph Traversals and Structure
# ----------------------------------------------------

def breadth_first_search(adj_list: Dict[int, List[int]], start: int) -> List[int]:
    """BFS traversal order starting from vertex start."""
    visited = {start}
    queue = collections.deque([start])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj_list.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return order


def depth_first_search(adj_list: Dict[int, List[int]], start: int) -> List[int]:
    """DFS traversal order starting from vertex start."""
    visited = set()
    order = []

    def dfs(u: int):
        visited.add(u)
        order.append(u)
        for v in adj_list.get(u, []):
            if v not in visited:
                dfs(v)

    dfs(start)
    return order


def connected_components_undirected(adj_list: Dict[int, List[int]], vertices: List[int]) -> List[List[int]]:
    """Find all connected components in an undirected graph."""
    visited = set()
    components = []
    for v in vertices:
        if v not in visited:
            comp = breadth_first_search(adj_list, v)
            visited.update(comp)
            components.append(comp)
    return components


def is_bipartite(adj_list: Dict[int, List[int]], vertices: List[int]) -> Tuple[bool, Dict[int, int]]:
    """Determine if graph is bipartite using 2-coloring. Returns (is_bipartite, color_map)."""
    color: Dict[int, int] = {}
    for node in vertices:
        if node not in color:
            color[node] = 0
            queue = collections.deque([node])
            while queue:
                u = queue.popleft()
                for v in adj_list.get(u, []):
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, {}
    return True, color


def has_cycle_undirected(adj_list: Dict[int, List[int]], vertices: List[int]) -> bool:
    """Check if undirected graph contains any cycles."""
    visited = set()

    def dfs(u: int, parent: Optional[int]) -> bool:
        visited.add(u)
        for v in adj_list.get(u, []):
            if v not in visited:
                if dfs(v, u):
                    return True
            elif v != parent:
                return True
        return False

    for v in vertices:
        if v not in visited:
            if dfs(v, None):
                return True
    return False


def has_cycle_directed(adj_list: Dict[int, List[int]], vertices: List[int]) -> bool:
    """Check if directed graph contains any cycles (using 3-color DFS)."""
    # 0 = unvisited, 1 = visiting, 2 = visited
    state = {v: 0 for v in vertices}

    def dfs(u: int) -> bool:
        state[u] = 1
        for v in adj_list.get(u, []):
            if state.get(v, 0) == 1:
                return True
            elif state.get(v, 0) == 0:
                if dfs(v):
                    return True
        state[u] = 2
        return False

    for v in vertices:
        if state[v] == 0:
            if dfs(v):
                return True
    return False


def topological_sort_kahn(adj_list: Dict[int, List[int]], vertices: List[int]) -> List[int]:
    """Kahn's in-degree topological sort algorithm for DAG."""
    in_degree = {v: 0 for v in vertices}
    for u in vertices:
        for v in adj_list.get(u, []):
            in_degree[v] = in_degree.get(v, 0) + 1
    queue = collections.deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(order) != len(vertices):
        raise ValueError("Graph has directed cycle; cannot topologically sort")
    return order


def tarjan_strongly_connected_components(adj_list: Dict[int, List[int]], vertices: List[int]) -> List[List[int]]:
    """Tarjan's algorithm for Strongly Connected Components (SCC) in directed graphs."""
    idx = 0
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    on_stack = set()
    stack = []
    sccs = []

    def strongconnect(u: int):
        nonlocal idx
        indices[u] = idx
        lowlink[u] = idx
        idx += 1
        stack.append(u)
        on_stack.add(u)

        for v in adj_list.get(u, []):
            if v not in indices:
                strongconnect(v)
                lowlink[u] = min(lowlink[u], lowlink[v])
            elif v in on_stack:
                lowlink[u] = min(lowlink[u], indices[v])

        if lowlink[u] == indices[u]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == u:
                    break
            sccs.append(scc)

    for v in vertices:
        if v not in indices:
            strongconnect(v)
    return sccs


# ----------------------------------------------------
# 4. Shortest Path Algorithms
# ----------------------------------------------------

def dijkstra_shortest_paths(adj_list: Dict[int, List[Tuple[int, float]]],
                            start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Dijkstra's single-source shortest path algorithm with non-negative weights."""
    dist = {start: 0.0}
    prev: Dict[int, Optional[int]] = {start: None}
    pq = [(0.0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj_list.get(u, []):
            if d + w < dist.get(v, float("inf")):
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, prev


def bellman_ford_shortest_paths(edges: List[Tuple[int, int, float]],
                                vertices: List[int],
                                start: int) -> Tuple[Dict[int, float], bool]:
    """Bellman-Ford algorithm. Returns (distances, has_negative_cycle)."""
    dist = {v: float("inf") for v in vertices}
    dist[start] = 0.0
    n = len(vertices)

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # Negative cycle check
    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            return dist, True  # Negative cycle detected
    return dist, False


def floyd_warshall_all_pairs(adj_matrix: List[List[float]]) -> List[List[float]]:
    """Floyd-Warshall all-pairs shortest paths O(V^3)."""
    n = len(adj_matrix)
    dist = [[adj_matrix[i][j] for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


def a_star_search(adj_list: Dict[int, List[Tuple[int, float]]],
                  start: int, goal: int,
                  heuristic: Callable[[int, int], float]) -> Optional[List[int]]:
    """A* heuristic shortest path search."""
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    came_from: Dict[int, int] = {}
    open_set = [(f_score[start], start)]
    open_set_hash = {start}

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for neighbor, weight in adj_list.get(current, []):
            tentative_g = g_score[current] + weight
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f, neighbor))
                    open_set_hash.add(neighbor)
    return None


# ----------------------------------------------------
# 5. Minimum Spanning Trees (MST)
# ----------------------------------------------------

def kruskal_minimum_spanning_tree(edges: List[Tuple[int, int, float]],
                                  vertices: List[int]) -> Tuple[List[Tuple[int, int, float]], float]:
    """Kruskal's MST algorithm with Disjoint Set Union."""
    sorted_edges = sorted(edges, key=lambda x: x[2])
    dsu = DisjointSetUnion(vertices)
    mst = []
    total_weight = 0.0
    for u, v, w in sorted_edges:
        if dsu.union(u, v):
            mst.append((u, v, w))
            total_weight += w
            if len(mst) == len(vertices) - 1:
                break
    return mst, total_weight


def prim_minimum_spanning_tree(adj_list: Dict[int, List[Tuple[int, float]]],
                               start: int) -> Tuple[List[Tuple[int, int, float]], float]:
    """Prim's MST algorithm using min-priority queue."""
    visited = {start}
    pq: List[Tuple[float, int, int]] = []
    for v, w in adj_list.get(start, []):
        heapq.heappush(pq, (w, start, v))
    mst = []
    total_weight = 0.0
    while pq:
        w, u, v = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        mst.append((u, v, w))
        total_weight += w
        for next_v, next_w in adj_list.get(v, []):
            if next_v not in visited:
                heapq.heappush(pq, (next_w, v, next_v))
    return mst, total_weight


# ----------------------------------------------------
# 6. Network Flow and Matching
# ----------------------------------------------------

def edmonds_karp_max_flow(capacity: List[List[float]], source: int, sink: int) -> Tuple[float, List[List[float]]]:
    """Edmonds-Karp (BFS-based Ford-Fulkerson) maximum network flow."""
    n = len(capacity)
    flow = [[0.0] * n for _ in range(n)]
    max_flow = 0.0

    while True:
        parent = [-1] * n
        parent[source] = source
        queue = collections.deque([source])
        bottleneck = [float("inf")] * n

        while queue and parent[sink] == -1:
            u = queue.popleft()
            for v in range(n):
                residual = capacity[u][v] - flow[u][v]
                if residual > 1e-12 and parent[v] == -1:
                    parent[v] = u
                    bottleneck[v] = min(bottleneck[u], residual)
                    queue.append(v)

        if parent[sink] == -1:
            break

        push = bottleneck[sink]
        max_flow += push
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += push
            flow[v][u] -= push
            v = u

    return max_flow, flow


# ----------------------------------------------------
# 7. Centralities and Metrics
# ----------------------------------------------------

def pagerank(adj_list: Dict[int, List[int]], vertices: List[int],
             damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[int, float]:
    """PageRank power iteration algorithm."""
    n = len(vertices)
    if n == 0:
        return {}
    rank = {v: 1.0 / n for v in vertices}
    out_degree = {v: len(adj_list.get(v, [])) for v in vertices}

    for _ in range(max_iter):
        new_rank = {v: (1.0 - damping) / n for v in vertices}
        dangling_sum = sum(rank[v] for v in vertices if out_degree[v] == 0)
        dangling_contrib = damping * dangling_sum / n

        for u in vertices:
            deg = out_degree[u]
            if deg > 0:
                share = damping * rank[u] / deg
                for v in adj_list.get(u, []):
                    new_rank[v] += share

        for v in vertices:
            new_rank[v] += dangling_contrib

        diff = sum(abs(new_rank[v] - rank[v]) for v in vertices)
        rank = new_rank
        if diff < tol:
            break
    return rank


def degree_centrality(adj_list: Dict[int, List[int]], vertices: List[int]) -> Dict[int, float]:
    """Degree centrality normalized by N - 1."""
    n = len(vertices)
    if n <= 1:
        return {v: 0.0 for v in vertices}
    return {v: len(adj_list.get(v, [])) / (n - 1.0) for v in vertices}


def graph_diameter_unweighted(adj_list: Dict[int, List[int]], vertices: List[int]) -> int:
    """Diameter (longest shortest path) of an unweighted graph."""
    max_d = 0
    for start in vertices:
        dist = {start: 0}
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d > max_d:
                max_d = d
            for v in adj_list.get(u, []):
                if v not in dist:
                    dist[v] = d + 1
                    queue.append(v)
    return max_d
def is_eulerian_undirected(adj_list: Dict[int, List[int]], vertices: List[int]) -> int:
    """Return 2 if Eulerian circuit exists (all even degree), 1 if Eulerian trail exists (exactly 2 odd degree), 0 otherwise."""
    odd_degrees = sum(1 for v in vertices if len(adj_list.get(v, [])) % 2 == 1)
    if odd_degrees == 0:
        return 2
    elif odd_degrees == 2:
        return 1
    return 0


def hierholzer_eulerian_circuit(adj_list: Dict[int, List[int]], start: int) -> List[int]:
    """Hierholzer's linear time algorithm to find Eulerian circuit in connected Eulerian graph."""
    adj_copy = {u: list(neighbors) for u, neighbors in adj_list.items()}
    curr_path = [start]
    circuit = []

    while curr_path:
        curr_v = curr_path[-1]
        if adj_copy.get(curr_v):
            next_v = adj_copy[curr_v].pop()
            # Remove reverse edge for undirected
            if curr_v in adj_copy.get(next_v, []):
                adj_copy[next_v].remove(curr_v)
            curr_path.append(next_v)
        else:
            circuit.append(curr_path.pop())
    return list(reversed(circuit))


def closeness_centrality(adj_list: Dict[int, List[int]], vertices: List[int]) -> Dict[int, float]:
    """Closeness centrality: (N - 1) / sum(d(u, v))."""
    n = len(vertices)
    if n <= 1:
        return {v: 0.0 for v in vertices}
    closeness = {}
    for u in vertices:
        dist = {u: 0}
        q = collections.deque([u])
        while q:
            curr = q.popleft()
            d = dist[curr]
            for neighbor in adj_list.get(curr, []):
                if neighbor not in dist:
                    dist[neighbor] = d + 1
                    q.append(neighbor)
        total_dist = sum(dist.values())
        closeness[u] = ((len(dist) - 1) / total_dist) * ((len(dist) - 1) / (n - 1)) if total_dist > 0 else 0.0
    return closeness


def betweenness_centrality_brandes(adj_list: Dict[int, List[int]], vertices: List[int]) -> Dict[int, float]:
    """Brandes' fast algorithm for betweenness centrality O(V*E)."""
    cb = {v: 0.0 for v in vertices}
    for s in vertices:
        stack = []
        predecessors: Dict[int, List[int]] = {w: [] for w in vertices}
        sigma = {w: 0 for w in vertices}
        sigma[s] = 1
        d = {w: -1 for w in vertices}
        d[s] = 0
        q = collections.deque([s])

        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj_list.get(v, []):
                if d[w] < 0:
                    q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta = {w: 0.0 for w in vertices}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    # Halve for undirected
    n = len(vertices)
    scale = 0.5 / max(1.0, ((n - 1) * (n - 2) * 0.5))
    return {v: cb[v] * scale for v in vertices}


def local_clustering_coefficient(adj_list: Dict[int, List[int]], u: int) -> float:
    """Local clustering coefficient of vertex u: 2 * E_neighbors / (k * (k - 1))."""
    neighbors = set(adj_list.get(u, []))
    k = len(neighbors)
    if k < 2:
        return 0.0
    edges_between = 0
    for v in neighbors:
        for w in adj_list.get(v, []):
            if w in neighbors:
                edges_between += 1
    return edges_between / (k * (k - 1))


def global_clustering_coefficient(adj_list: Dict[int, List[int]], vertices: List[int]) -> float:
    """Average clustering coefficient across all vertices."""
    if not vertices:
        return 0.0
    return sum(local_clustering_coefficient(adj_list, v) for v in vertices) / len(vertices)


def vertex_eccentricity(adj_list: Dict[int, List[int]], vertices: List[int], u: int) -> int:
    """Eccentricity: maximum shortest distance from vertex u to any reachable vertex."""
    dist = {u: 0}
    q = collections.deque([u])
    while q:
        curr = q.popleft()
        d = dist[curr]
        for v in adj_list.get(curr, []):
            if v not in dist:
                dist[v] = d + 1
                q.append(v)
    return max(dist.values()) if dist else 0


def graph_radius_unweighted(adj_list: Dict[int, List[int]], vertices: List[int]) -> int:
    """Graph radius: minimum eccentricity among all vertices."""
    return min(vertex_eccentricity(adj_list, vertices, v) for v in vertices) if vertices else 0


def graph_center_vertices(adj_list: Dict[int, List[int]], vertices: List[int]) -> List[int]:
    """Vertices whose eccentricity equals the graph radius."""
    rad = graph_radius_unweighted(adj_list, vertices)
    return [v for v in vertices if vertex_eccentricity(adj_list, vertices, v) == rad]


def graph_periphery_vertices(adj_list: Dict[int, List[int]], vertices: List[int]) -> List[int]:
    """Vertices whose eccentricity equals the graph diameter."""
    diam = graph_diameter_unweighted(adj_list, vertices)
    return [v for v in vertices if vertex_eccentricity(adj_list, vertices, v) == diam]


def greedy_vertex_coloring(adj_list: Dict[int, List[int]], vertices: List[int]) -> Dict[int, int]:
    """Greedy vertex coloring assigning smallest available color index."""
    color: Dict[int, int] = {}
    for u in vertices:
        neighbor_colors = {color[v] for v in adj_list.get(u, []) if v in color}
        c = 0
        while c in neighbor_colors:
            c += 1
        color[u] = c
    return color


def welsh_powell_coloring(adj_list: Dict[int, List[int]], vertices: List[int]) -> Dict[int, int]:
    """Welsh-Powell heuristic vertex coloring sorting by descending vertex degree."""
    sorted_vertices = sorted(vertices, key=lambda v: len(adj_list.get(v, [])), reverse=True)
    color: Dict[int, int] = {}
    current_color = 0

    while len(color) < len(vertices):
        for u in sorted_vertices:
            if u not in color:
                neighbor_colors = {color[v] for v in adj_list.get(u, []) if v in color}
                if current_color not in neighbor_colors:
                    color[u] = current_color
        current_color += 1
    return color


def is_valid_vertex_coloring(adj_list: Dict[int, List[int]], coloring: Dict[int, int]) -> bool:
    """Verify that no two adjacent vertices share the same color."""
    for u, neighbors in adj_list.items():
        for v in neighbors:
            if u in coloring and v in coloring and coloring[u] == coloring[v]:
                return False
    return True


def is_tree(adj_list: Dict[int, List[int]], vertices: List[int]) -> bool:
    """Check if undirected graph is a tree (connected and acyclic, |E| = |V| - 1)."""
    n = len(vertices)
    if n == 0:
        return True
    comps = connected_components_undirected(adj_list, vertices)
    if len(comps) != 1:
        return False
    # Count unique edges
    edge_count = sum(len(adj_list.get(v, [])) for v in vertices) // 2
    return edge_count == n - 1


def generate_complete_graph(n: int) -> Dict[int, List[int]]:
    """Generate complete graph K_n."""
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def generate_complete_bipartite_graph(n1: int, n2: int) -> Dict[int, List[int]]:
    """Generate complete bipartite graph K_{n1, n2}."""
    adj = {i: [] for i in range(n1 + n2)}
    for u in range(n1):
        for v in range(n1, n1 + n2):
            adj[u].append(v)
            adj[v].append(u)
    return adj


def generate_cycle_graph(n: int) -> Dict[int, List[int]]:
    """Generate cycle graph C_n."""
    if n < 3:
        raise ValueError("Cycle graph requires n >= 3")
    adj = {i: [] for i in range(n)}
    for i in range(n):
        adj[i] = [(i - 1) % n, (i + 1) % n]
    return adj


def generate_star_graph(n: int) -> Dict[int, List[int]]:
    """Generate star graph S_n with center vertex 0 and n-1 leaves."""
    adj = {i: [] for i in range(n)}
    for i in range(1, n):
        adj[0].append(i)
        adj[i].append(0)
    return adj


def generate_wheel_graph(n: int) -> Dict[int, List[int]]:
    """Generate wheel graph W_n with hub 0 and cycle on 1..n-1."""
    k = n - 1
    adj = {i: [] for i in range(n)}
    for i in range(1, n):
        adj[0].append(i)
        adj[i].append(0)
        adj[i].append(1 + (i - 1 - 1) % k)
        adj[i].append(1 + (i - 1 + 1) % k)
    return adj


def generate_path_graph(n: int) -> Dict[int, List[int]]:
    """Generate path graph P_n (0 - 1 - 2 - ... - n-1)."""
    adj = {i: [] for i in range(n)}
    for i in range(n - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    return adj


def generate_petersen_graph() -> Dict[int, List[int]]:
    """Generate standard 10-vertex 3-regular Petersen graph."""
    adj = {i: [] for i in range(10)}
    # Outer 5-cycle (0-4)
    for i in range(5):
        adj[i].append((i + 1) % 5)
        adj[i].append((i - 1) % 5)
    # Inner 5-star (5-9)
    for i in range(5):
        adj[5 + i].append(5 + (i + 2) % 5)
        adj[5 + i].append(5 + (i - 2) % 5)
    # Radial spokes connecting outer to inner
    for i in range(5):
        adj[i].append(5 + i)
        adj[5 + i].append(i)
    return adj


def graph_density(num_vertices: int, num_edges: int, directed: bool = False) -> float:
    """Graph edge density ratio |E| / max_possible_edges."""
    if num_vertices <= 1:
        return 0.0
    max_edges = num_vertices * (num_vertices - 1) if directed else num_vertices * (num_vertices - 1) * 0.5
    return num_edges / max_edges


def graph_laplacian_matrix(adj_matrix: List[List[float]]) -> List[List[float]]:
    """Construct combinatorial graph Laplacian L = D - A."""
    n = len(adj_matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = sum(adj_matrix[i])
        L[i][i] = deg
        for j in range(n):
            if i != j:
                L[i][j] = -adj_matrix[i][j]
    return L
def graph_degree_sequence(adj_list: Dict[int, List[int]], vertices: List[int]) -> List[int]:
    """Degree sequence of graph sorted in non-increasing order."""
    return sorted([len(adj_list.get(v, [])) for v in vertices], reverse=True)


def is_graphical_havel_hakimi(degrees: List[int]) -> bool:
    """Check if degree sequence is realizable by a simple graph via Havel-Hakimi theorem."""
    d = sorted([deg for deg in degrees if deg >= 0], reverse=True)
    while d:
        if d[0] == 0:
            return True
        v = d.pop(0)
        if v > len(d):
            return False
        for i in range(v):
            d[i] -= 1
            if d[i] < 0:
                return False
        d.sort(reverse=True)
    return True


def graph_is_regular(adj_list: Dict[int, List[int]], vertices: List[int]) -> Tuple[bool, Optional[int]]:
    """Check if graph is k-regular (all vertices have identical degree)."""
    if not vertices:
        return True, 0
    k = len(adj_list.get(vertices[0], []))
    for v in vertices:
        if len(adj_list.get(v, [])) != k:
            return False, None
    return True, k


def has_hamiltonian_path(adj_list: Dict[int, List[int]], vertices: List[int]) -> bool:
    """Backtracking verification of whether graph contains a Hamiltonian path."""
    n = len(vertices)
    visited = set()

    def backtrack(u: int, count: int) -> bool:
        if count == n:
            return True
        visited.add(u)
        for v in adj_list.get(u, []):
            if v not in visited:
                if backtrack(v, count + 1):
                    return True
        visited.remove(u)
        return False

    for start in vertices:
        if backtrack(start, 1):
            return True
    return False


def tree_diameter(tree_adj: Dict[int, List[int]], vertices: List[int]) -> int:
    """Diameter of tree using double-BFS method in O(V) time."""
    if not vertices:
        return 0
    def bfs_farthest(start: int) -> Tuple[int, int]:
        dist = {start: 0}
        q = collections.deque([start])
        farthest_node = start
        while q:
            u = q.popleft()
            d = dist[u]
            if d > dist[farthest_node]:
                farthest_node = u
            for v in tree_adj.get(u, []):
                if v not in dist:
                    dist[v] = d + 1
                    q.append(v)
        return farthest_node, dist[farthest_node]

    u, _ = bfs_farthest(vertices[0])
    v, diam = bfs_farthest(u)
    return diam


def generate_hypercube_graph(d: int) -> Dict[int, List[int]]:
    """Generate d-dimensional hypercube graph Q_d with 2^d vertices."""
    num_nodes = 1 << d
    adj = {i: [] for i in range(num_nodes)}
    for u in range(num_nodes):
        for bit in range(d):
            v = u ^ (1 << bit)
            adj[u].append(v)
    return adj


def generate_erdos_renyi_graph(n: int, p: float, seed: int = 42) -> Dict[int, List[int]]:
    """Generate random Erdos-Renyi G(n, p) graph with deterministic linear congruential PRNG."""
    adj = {i: [] for i in range(n)}
    state = seed
    for i in range(n):
        for j in range(i + 1, n):
            state = (1103515245 * state + 12345) & 0x7fffffff
            u = state / 2147483648.0
            if u < p:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def hopcroft_karp_bipartite_matching(bipartite_adj: Dict[int, List[int]],
                                     left_nodes: List[int], right_nodes: List[int]) -> Dict[int, int]:
    """Hopcroft-Karp maximum bipartite matching O(E * sqrt(V))."""
    pair_u = {u: None for u in left_nodes}
    pair_v = {v: None for v in right_nodes}
    dist: Dict[Optional[int], int] = {}

    def bfs() -> bool:
        queue = collections.deque()
        for u in left_nodes:
            if pair_u[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float("inf")
        dist[None] = float("inf")
        while queue:
            u = queue.popleft()
            if dist[u] < dist[None]:
                for v in bipartite_adj.get(u, []):
                    next_u = pair_v[v]
                    if dist.get(next_u, float("inf")) == float("inf"):
                        dist[next_u] = dist[u] + 1
                        queue.append(next_u)
        return dist[None] != float("inf")

    def dfs(u: Optional[int]) -> bool:
        if u is not None:
            for v in bipartite_adj.get(u, []):
                next_u = pair_v[v]
                if dist.get(next_u, float("inf")) == dist[u] + 1:
                    if dfs(next_u):
                        pair_v[v] = u
                        pair_u[u] = v
                        return True
            dist[u] = float("inf")
            return False
        return True

    while bfs():
        for u in left_nodes:
            if pair_u[u] is None:
                dfs(u)

    return {u: pair_u[u] for u in left_nodes if pair_u[u] is not None}


def prufer_sequence_from_tree(tree_adj: Dict[int, List[int]], vertices: List[int]) -> List[int]:
    """Prufer sequence of labelled tree on vertices 0..n-1."""
    n = len(vertices)
    if n <= 2:
        return []
    adj_copy = {u: set(tree_adj.get(u, [])) for u in vertices}
    deg = {u: len(adj_copy[u]) for u in vertices}
    leaves = [u for u in vertices if deg[u] == 1]
    heapq.heapify(leaves)
    seq = []

    for _ in range(n - 2):
        leaf = heapq.heappop(leaves)
        neighbor = next(iter(adj_copy[leaf]))
        seq.append(neighbor)
        adj_copy[neighbor].remove(leaf)
        deg[neighbor] -= 1
        if deg[neighbor] == 1:
            heapq.heappush(leaves, neighbor)
    return seq


def tree_from_prufer_sequence(sequence: List[int]) -> Dict[int, List[int]]:
    """Reconstruct tree adjacency list from length n-2 Prufer sequence."""
    n = len(sequence) + 2
    deg = {i: 1 for i in range(n)}
    for x in sequence:
        deg[x] += 1
    leaves = [i for i in range(n) if deg[i] == 1]
    heapq.heapify(leaves)
    tree_adj = {i: [] for i in range(n)}

    for x in sequence:
        leaf = heapq.heappop(leaves)
        tree_adj[leaf].append(x)
        tree_adj[x].append(leaf)
        deg[x] -= 1
        if deg[x] == 1:
            heapq.heappush(leaves, x)

    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    tree_adj[u].append(v)
    tree_adj[v].append(u)
    return tree_adj


def algebraic_connectivity_fiedler_bound(laplacian: List[List[float]]) -> float:
    """Lower bound approximation on second-smallest eigenvalue (algebraic connectivity) of Laplacian."""
    n = len(laplacian)
    if n <= 1:
        return 0.0
    trace = sum(laplacian[i][i] for i in range(n))
    return 2.0 * trace / (n * (n - 1))


def graph_is_connected(adj_list: Dict[int, List[int]], vertices: List[int]) -> bool:
    """Check if undirected graph is connected."""
    if not vertices:
        return True
    return len(connected_components_undirected(adj_list, vertices)) == 1
