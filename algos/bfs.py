# Graph example: https://www.geeksforgeeks.org/wp-content/uploads/bfs11.png

graph = {1: set([2, 3]),
         2: set([1, 4, 5]),
         3: set([1, 5]),
         4: set([2, 5, 6]),
         5: set([2, 3, 4, 6]),
         6: set([4, 5])}

def bfs(graph, start):
    visited = []
    q = [start]
    while q:
        vertex = q.pop(0)
        if vertex not in visited:
            visited.append(vertex)
            for i in graph[vertex]:
                if i not in visited:
                    q.append(i)
    return visited

visits = bfs(graph, 1)
print(visits)