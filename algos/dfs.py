# Graph example https://www.geeksforgeeks.org/wp-content/uploads/DFS9.png

graph = {'A': set(["B", "C"]),
         'B': set(["A", "D", "E"]),
         'C': set(["A", "E"]),
         'D': set(["B", "E", "F"]),
         'E': set(["B", "C", "D", "F"]),
         'F': set(["D", "E"])}

def dfs(graph, start):
    visited = []
    stack = [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            for i in graph[vertex]:
                if i not in visited:
                    stack.append(i)
    return visited

# NOTE: due to the randomness of selecting from set in the for loop,
# order may vary for each run
visits = dfs(graph, "A")
print(visits)