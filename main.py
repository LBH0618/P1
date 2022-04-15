# INFS7450 Social Media Analytics
# Project 1 – Fast Computation of User Centrality
# Banghong Liang
# s4633687
# 2022/04/06
import networkx as nx
import numpy as np
import matplotlib.pyplot


def betweenness_centrality_calculation(graph):
    """calculation for betweenness centrality"""
    return


# def pagerank_centrality_calculation():
#  calculation for pagerank centrality


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

    # Graph Visualization
    print(facebook_graph)
    # print(facebook_graph[0])
    nx.draw(facebook_graph, with_labels=True)
    matplotlib.pyplot.show()

    # Betweenness Centrality Calculation
    # nx.betweenness_centrality(facebook_graph)
    # print(nx.betweenness_centrality(facebook_graph))
    betweenness_centrality_calculation(facebook_graph)

    # Pagerank Centrality Calculation
    # alpha = 0.85, beta = 0.15
    nx.pagerank(facebook_graph)
    # print(nx.pagerank(facebook_graph))
