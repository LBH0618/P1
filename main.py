# INFS7450 Social Media Analytics
# Project 1 – Fast Computation of User Centrality
# Banghong Liang
# 46336873
# 2022/04/06
from collections import deque
from heapq import nlargest
import networkx as nx
import matplotlib.pyplot


def betweenness_centrality_calculation(graph):
    """
    calculation for betweenness centrality.
    :param graph: networks
    :return: dictionary {node,betweenness centrality}
    """
    if len(graph) == 0:
        return {}
    # Initialization
    dict_node_value = dict.fromkeys(graph, 0.0)
    # Iterate through each vertex
    for single_source in dict_node_value:
        # Breadth first search for shortest path
        visited_vertices, predecessors, sigma, distance = _breadth_first_search(graph, single_source)
        # Accumulate the values -> back-propagation of dependencies
        dict_node_value = _accumulate(dict_node_value, visited_vertices, predecessors, sigma, single_source)
    # return dictionary for node's betweenness centrality
    return dict_node_value


def _breadth_first_search(graph, source):
    """ Using breadth first search to find the shortest path for the single-source
    :param graph: graph for facebook networks
    :param source: single source
    :return: visited_vertices, predecessors, sigma, distance
    """
    # Initialization
    predecessors = {}  # list of predecessors on shortest paths from source
    for vertex in graph:
        predecessors[vertex] = []
    sigma = dict.fromkeys(graph, 0.0)
    sigma[source] = 1.0  # sigma[s] <- 1
    queue_bfs = deque([source])  # enqueue s -> queue
    visited_vertices = []
    distance = {}
    while queue_bfs:
        vertex = queue_bfs.popleft()
        if vertex == source:
            distance[vertex] = 0  # dist[s] <- 0
        visited_vertices.append(vertex)
        for vertex_w in graph[vertex]:
            if vertex_w not in distance:  # if dist[w] = infinite then
                distance[vertex_w] = distance[vertex] + 1  # dist[w] <- dist[v] + 1
                queue_bfs.append(vertex_w)
            if distance[vertex_w] == distance[vertex] + 1:
                sigma[vertex_w] += sigma[vertex]  # sigma[w] <- sigma[w] + sigma[vertex]
                predecessors[vertex_w].append(vertex)  # append vertex -> predecessors[w]
    return visited_vertices, predecessors, sigma, distance


def _accumulate(dict_node_value, visited_vertices, predecessors, sigma, source):
    """ Back-propagation of dependencies.
    :param dict_node_value: dictionary for the graph
    :param visited_vertices: list of visited vertices
    :param predecessors: list of predecessors
    :param sigma: number of shortest paths
    :param source: source
    :return: dict_node_value after accumulation
    """
    # dependence of source
    dependency = dict.fromkeys(visited_vertices, 0)  # dependency[]
    while visited_vertices:
        vertex_w = visited_vertices.pop()  # pop w <- visited_vertices
        for vertex in predecessors[vertex_w]:
            # dependency[v] + sigma[v]/sigma[w] * (1 + dependency[w])
            dependency[vertex] += sigma[vertex] / sigma[vertex_w] * (1 + dependency[vertex_w])
        if vertex_w != source:
            dict_node_value[vertex_w] += dependency[vertex_w] # Cb[w] + dependency[w]
    return dict_node_value


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
    num_nodes = dummy_graph.number_of_nodes() # number of nodes for the given graph
    # Normalized
    normalized_graph = dict.fromkeys(dummy_graph, 1.0 / num_nodes)
    # Power Iteration for Pagerank
    for i in range(100):
        normalized_graph_last = normalized_graph
        normalized_graph = dict.fromkeys(normalized_graph_last.keys(), 0)  # initialization
        for node in normalized_graph:
            for to_node in stochastic_graph[node]:
                normalized_graph[to_node] += alpha * normalized_graph_last[node] \
                                             * stochastic_graph[node][to_node]['weight']
            normalized_graph[node] += beta / num_nodes  # beta = 0.15
        threshold = 0
        for node in normalized_graph:
            threshold += abs(normalized_graph[node] - normalized_graph_last[node])
        if num_nodes * 0.0000001 > threshold:
            return normalized_graph


def print_top_key(graph, num):
    """ Print top 10 keys for the dictionary by the sorting the value of key
    :param graph: networks
    :param num: number of keys wants to print
    """
    dummy_graph = graph
    top_10 = nlargest(num, dummy_graph, key=dummy_graph.get)
    for node in top_10:
        print(node, end=" ")
    print()


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
    q1_result = betweenness_centrality_calculation(facebook_graph)

    # Pagerank Centrality Calculation #
    # alpha = 0.85, beta = 0.15
    # nx_q2_result = nx.pagerank(facebook_graph)
    q2_result = pagerank_centrality_calculation(facebook_graph, 0.85, 0.15)

    # Result Stream Out #
    # print(facebook_graph)
    # Betweenness
    # print_top_key(nx_q1_result, 10)
    print_top_key(q1_result, 10)
    # Pagerank
    # print_top_key(nx_q2_result, 10)
    print_top_key(q2_result, 10)
