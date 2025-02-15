from collections import deque

def bfs(graph, start_node):
    # Create a queue (FIFO)
    queue = deque([start_node])
    
    # Dictionary to keep track of visited nodes
    visited = {node: False for node in graph}
    visited[start_node] = True

    while queue:
        # Remove the first node from the queue
        node = queue.popleft()
        
        # Process the node (e.g., print it)
        print(node, end=" ")

        # Explore unvisited neighbors
        for neighbor in graph[node]:
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True

# Define the graph using an adjacency list representation
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Execute BFS starting from node 'A'
bfs(graph, 'A')