import numpy as np
import heapq

"""
Graph below represents: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#/media/File:Dijkstra_Animation.gif
"""

graph = {
    "1": {"2": 7, "3": 9, "6": 14},
    "2": {"1": 7, "3": 10, "4": 15},
    "3": {"1": 9, "2": 10, "4": 11, "6": 2},
    "4": {"2": 15, "3": 11, "6": 2},
    "5": {"4": 6, "6": 9},
    "6": {"1": 14, "3": 2, "5": 9}
}

def dijsktra(graph, source):
    dist = {}
    visited = []
    h = []
    for i in graph:
        if i == source:
            dist[i] = 0
            heapq.heappush(h, (0, source))
        else:
            dist[i] = np.inf
    while h:
        node_heap = heapq.heappop(h)
        u = node_heap[1]
        dist_to_u = node_heap[0]
        for v in graph[u]:
            alt_dist = dist[u] + graph[u][v]
            if dist[v] > alt_dist:
                dist[v] = alt_dist
            if v not in visited:
                heapq.heappush(h, (dist[v], v))
        visited.append(u)
    return dist

if __name__ == "__main__":
    dist = dijsktra(graph, "1")
    assert {'1': 0, '2': 7, '3': 9, '4': 20, '5': 20, '6': 11} == dist