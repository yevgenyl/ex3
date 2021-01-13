import unittest
import networkx as nx
from src.GraphAlgo import GraphAlgo
import time


class TestCompareToNetworkX(unittest.TestCase):

    def test_shortest_path(self):
        # _________________ Test My DiGraph shortest path _________________

        my_graph_algo = GraphAlgo()
        my_graph_algo.load_from_json("../Graphs_no_pos/G_10_80_0.json")
        my_graph = my_graph_algo.g
        start = time.time()
        print(str(my_graph_algo.shortest_path(0, 8)))
        end = time.time()
        print("Dijkstra - My DiGraph time: " + str(end - start))

        # _________________ Test NetworkX DiGraph shortest path _________________

        networkx_graph = nx.DiGraph()
        for node in my_graph.get_all_v().values():
            networkx_graph.add_node(node.key)
        for src in my_graph.E.keys():
            for dest, weight in my_graph.E[src].items():
                networkx_graph.add_edge(src, dest, weight=weight)
        start = time.time()
        print(str(nx.single_source_dijkstra(networkx_graph, 0, 8)))
        end = time.time()
        print("Dijkstra - NetworkX DiGraph time: " + str(end - start))

    def test_connected_components(self):
        # _________________ Test My DiGraph components _________________

        my_graph_algo = GraphAlgo()
        my_graph_algo.load_from_json("../Graphs_no_pos/G_30000_240000_0.json")
        my_graph = my_graph_algo.g
        start = time.time()
        my_graph_algo.connected_components()
        end = time.time()
        print("SCC - My DiGraph time: " + str(end - start))

        # _________________ Test NetworkX DiGraph components _________________

        networkx_graph = nx.DiGraph()
        for node in my_graph.get_all_v().values():
            networkx_graph.add_node(node.key)
        for src in my_graph.E.keys():
            for dest, weight in my_graph.E[src].items():
                networkx_graph.add_edge(src, dest, weight=weight)
        start = time.time()
        nx.strongly_connected_components(networkx_graph)
        end = time.time()
        print("SCC - NetworkX DiGraph time: " + str(end - start))
        # for i in componenets:
        #     print(i, end=", ")
        # print("SCC - NetworkX DiGraph time: " + str(end - start))

    def test_connected_component(self):
        # _________________ Test My DiGraph component _________________
        my_graph_algo = GraphAlgo()
        my_graph_algo.load_from_json("../Graphs_no_pos/G_30000_240000_0.json")
        my_graph = my_graph_algo.g
        start = time.time()
        my_graph_algo.connected_component(5)
        end = time.time()
        print("Component - My DiGraph time: " + str(end - start))


if __name__ == '__main__':
    unittest.main()
