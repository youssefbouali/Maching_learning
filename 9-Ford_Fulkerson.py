from collections import deque

def bfs(graph, source, sink, parent):
    """
    Perform BFS to find an augmenting path in the residual graph.
    """
    visited = [False] * len(graph)
    queue = deque([source])
    visited[source] = True

    while queue:
        u = queue.popleft()

        for v, capacity in enumerate(graph[u]):
            if not visited[v] and capacity > 0:  # If capacity is available
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False

def ford_fulkerson(graph, source, sink):
    """
    Compute the maximum flow from source to sink using Ford-Fulkerson method.
    """
    parent = [-1] * len(graph)
    max_flow = 0
    residual_graph = [row[:] for row in graph]  # Make a copy of the graph

    while bfs(residual_graph, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s])
            s = parent[s]

        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = parent[v]

        max_flow += path_flow

    return max_flow

# Example usage
graph = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]

source, sink = 0, 5
print("The maximum possible flow is", ford_fulkerson(graph, source, sink))