def dijkstra(graph, source):
    """
    Compute the shortest paths from the source node to all other nodes using Dijkstra's algorithm
    without a priority queue.

    :param graph: Dictionary where keys are nodes and values are lists of (neighbor, weight) tuples.
    :param source: The starting node.
    :return: Dictionary of shortest distances from source to each node.
    """
    # Initialize distances to infinity for all nodes
    dist = {node: float('inf') for node in graph}
    dist[source] = 0  # Distance to the source is 0

    # Set of unvisited nodes
    unvisited = set(graph.keys())

    while unvisited:
        # Select the unvisited node with the smallest known distance
        current_node = min(unvisited, key=lambda node: dist[node])

        # If the smallest distance is infinity, we stop (remaining nodes are unreachable)
        if dist[current_node] == float('inf'):
            break

        # Remove the current node from unvisited set
        unvisited.remove(current_node)

        # Update distances for neighbors
        for neighbor, weight in graph[current_node]:
            if neighbor in unvisited:
                new_distance = dist[current_node] + weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance

    return dist

# Example graph representation (adjacency list)
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

# Running Dijkstra's algorithm from source node 'A'
source_node = 'A'
shortest_paths = dijkstra(graph, source_node)

# Print shortest paths from the source
print(f"Shortest distances from node {source_node}:")
for node, distance in shortest_paths.items():
    print(f"{node}: {distance}")