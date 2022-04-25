# INFS7450 Social Media Analytics
# Project 1 – Fast Computation of User Centrality
# Banghong Liang
# s4633687
# 2022/04/06
from collections import deque

import networkx as nx
import numpy as np
import matplotlib.pyplot


def betweenness_centrality_calculation(graph):
    """
    calculation for betweenness centrality.
    :param graph: networks
    :return: dictionary {node,betweenness centrality}
    """
    if len(graph) == 0:
        return {}
    # Create dictionary for graph with
    # key - node
    # value - betweenness centrality for the node
    dict_node_value = dict.fromkeys(graph, 0.0)
    # Create a dummy graph to work with
    dummy_graph = graph
    # Iterate through each vertex
    for s in dummy_graph:
        # Breadth first search for shortest path
        S, P, sigma, _ = _breadth_first_search(graph, s)
        # Accumulate the values
        dict_node_value, delta = _accumulate(dict_node_value, S, P, sigma, s)
    # return dictionary for node's betweenness centrality
    return dict_node_value


def _breadth_first_search(graph, s):
    """
    :param graph: networks
    :param s:
    :return:
    """
    S = []  # list
    P = {}  # dictionary

    # Initialization
    for vertex in graph:
        P[vertex] = []
    sigma = dict.fromkeys(graph, 0.0)
    D = {s: 0}  # dictionary
    sigma[s] = 1.0  # visited
    queue_bfs = deque([s])
    while queue_bfs:
        vertex = queue_bfs.popleft()
        S.append(vertex)
        d_vertex = D[vertex]
        sigma_vertex = sigma[vertex]
        for w in graph[vertex]:
            if w not in D:
                queue_bfs.append(w)
                D[w] = d_vertex + 1
            if D[w] == d_vertex + 1:
                sigma[w] += sigma_vertex
                P[w].append(vertex)
    return S, P, sigma, D


def _accumulate(dict_node_value, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        betweenness_coefficient = (1 + delta[w]) / sigma[w]
        for vertex in P[w]:
            delta[vertex] += sigma[vertex] * betweenness_coefficient
        if w != s:
            dict_node_value[w] += delta[w]
    return dict_node_value, delta


def pagerank_centrality_calculation(graph, alpha, beta):
    """
    Based on the structure of the inbound links, PageRank ranks the nodes in the graph.
    Calculation for pagerank centrality.f
    :param graph: networks
    :param alpha: coefficient = 0.85 by project 1 requirements
    :param beta: coefficient = 0.15 by project 1 requirements
    :return: probability distribution for pagerank in dictionary {node,pagerank centrality}
    """
    if len(graph) == 0:
        return {}
    # convert undirected graph to directed graph
    dummy_graph = graph.to_directed()
    # convert directed graph to right-stochastic graph
    # Normalize the edges for each node so that the sum of out edges of a node is equal to 1 (weight = 1).
    stochastic_graph = nx.stochastic_graph(dummy_graph)
    num_nodes = dummy_graph.number_of_nodes()
    # Normalized
    normalized_graph = dict.fromkeys(dummy_graph, 1.0 / num_nodes)

    # Power Iteration for Pagerank
    for i in range(1000):
        normalized_graph_last = normalized_graph
        normalized_graph = dict.fromkeys(normalized_graph_last.keys(), 0)  # initialization
        for node in normalized_graph:
            for to_node in stochastic_graph[node]:
                normalized_graph[to_node] += alpha * normalized_graph_last[node] \
                                             * stochastic_graph[node][to_node]['weight']
            normalized_graph[node] += beta / num_nodes  # beta = 1 - alpha
        threshold = 0
        for node in normalized_graph:
            threshold += abs(normalized_graph[node] - normalized_graph_last[node])
        # threshold = sum(abs(normalized_graph[node] - normalized_graph_last[node]) for node in normalized_graph)
        if num_nodes * 0.0000001 > threshold:
            return normalized_graph


def print_top_10(graph):
    graph_list = []
    for (node, value) in graph.items():
        graph_list.append((node, value))
    graph_list.sort(key=lambda x: x[1], reverse=True)
    print({k: v for (k, v) in graph_list[0:10]})


if __name__ == '__main__':
    facebook_graph = nx.Graph()
    # File reading from dataset
    file = open("3.data.txt", 'r')
    data = file.readlines()
    for row in data:
        row_data = row.split()
        # facebook_graph.add_node(int(row_data[0]))
        facebook_graph.add_edge(int(row_data[0]), int(row_data[1]))
    file.close()

    # Graph Visualization #
    # print(facebook_graph[0])
    # nx.draw(facebook_graph, with_labels=True)
    # matplotlib.pyplot.show()

    # Betweenness Centrality Calculation #
    # nx_q1_result = nx.betweenness_centrality(facebook_graph)
    # q1_result = betweenness_centrality_calculation(facebook_graph)

    # Pagerank Centrality Calculation #
    # alpha = 0.85, beta = 0.15
    nx_q2_result = nx.pagerank(facebook_graph)
    q2_result = pagerank_centrality_calculation(facebook_graph, 0.85, 0.15)

    # Result Stream Out #
    print(facebook_graph)
    # Betweenness
    # print(print_top_10(nx_q1_result))
    # print(print_top_10(q1_result))
    # Pagerank
    print_top_10(nx_q2_result)
    print_top_10(q2_result)
