import numpy as np
import heapq

"""
Graph below represents: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#/media/File:Dijkstra_Animation.gif
Negative edge weights are allowed
Negative weight cycles not allowed -> graph has no shortest distances
"""

graph = {
    "1": {"2": 7, "3": 9, "6": 14},
    "2": {"1": 7, "3": 10, "4": 15},
    "3": {"1": 9, "2": 10, "4": 11, "6": 2},
    "4": {"2": 15, "3": 11, "6": 2},
    "5": {"4": 6, "6": 9},
    "6": {"1": 14, "3": 2, "5": -5}
}

def bellman_ford(graph, source):
    dist = {}
    for i in graph:
        if i == source:
            dist[i] = 0
        else:
            dist[i] = np.inf
    for i in range(len(graph)):
        for u in graph:
            for v in graph[u]:
                alt_dist = dist[u] + graph[u][v]
                if dist[v] > alt_dist:
                    dist[v] = alt_dist
    # check for cycles
    for u in graph:
        for v in graph[u]:
            alt_dist = dist[u] + graph[u][v]
            if dist[v] > alt_dist:
                raise Exception("Negative cycle")
    return dist

if __name__ == "__main__":
    dist = bellman_ford(graph, "1")
    assert {'1': 0, '2': 7, '3': 9, '4': 12, '5': 6, '6': 11} == dist