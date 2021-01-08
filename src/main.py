from DiGraph import DiGraph
from GraphAlgo import GraphAlgo

if __name__ == '__main__':
    d = DiGraph()
    d.add_node(1)
    d.add_node(2)
    d.add_node(3)
    d.add_node(4)
    d.add_edge(1, 2, 0.45)
    d.add_edge(1, 3, 0.5)
    d.add_edge(3, 1, 0.2)
    d.add_edge(4, 1, 0.3)
    print(d)
    print("Node size: " + str(d.v_size()))
    print("Edge size: " + str(d.e_size()))
    print("All vertices: " + str(d.get_all_v()))
    print("All out edges of node: " + str(d.all_out_edges_of_node(3)))
    print("All in edges of node: " + str(d.all_in_edges_of_node(3)))
    print("MC: " + str(d.MC))
    print(str(d.E))
    print(str(d.R))
    d.remove_edge(3, 1)  # Remove one edge
    d.remove_edge(5, 6)  # Not exist
    d.remove_edge(1, 4)  # Not exist
    print(d)
    print("Node size: " + str(d.v_size()))
    print("Edge size: " + str(d.e_size()))
    print("All vertices: " + str(d.get_all_v()))
    print("All out edges of node: " + str(d.all_out_edges_of_node(3)))
    print("All in edges of node: " + str(d.all_in_edges_of_node(3)))
    print("MC: " + str(d.MC))
    print(str(d.E))
    print(str(d.R))
    d.remove_node(3)
    print(d)
    print("Node size: " + str(d.v_size()))
    print("Edge size: " + str(d.e_size()))
    print("All vertices: " + str(d.get_all_v()))
    print("All out edges of node: " + str(d.all_out_edges_of_node(3)))
    print("All in edges of node: " + str(d.all_in_edges_of_node(3)))
    print("MC: " + str(d.MC))
    print(str(d.E))
    print(str(d.R))

    '''
    GraphAlgo 
    '''
    d.get_all_v()[1].pos = (35.20500018886199, 32.10785303529412, 0.0)
    print("Nodes:*** "+str(d.V))
    print("Edges:*** "+str(d.E))
    algo = GraphAlgo()
    algo.g = d
    algo.save_to_json("graph.json")

    algo2 = GraphAlgo()
    algo2.load_from_json("graph.json")
    print(algo2.get_graph())

    graph_g = DiGraph()
    for i in range(1, 10):
        graph_g.add_node(i)
    graph_g.add_edge(1, 2, 1)
    graph_g.add_edge(1, 4, 30)
    graph_g.add_edge(1, 5, 400)
    graph_g.add_edge(2, 3, 1)
    graph_g.add_edge(3, 4, 1)
    graph_g.add_edge(5, 4, 1)
    graph_g.add_edge(4, 6, 1)
    graph_g.add_edge(5, 7, 1)
    graph_g.add_edge(3, 6, 50)
    graph_g.add_edge(7, 6, 1)
    graph_g.add_edge(6, 9, 1)
    graph_g.add_edge(6, 8, 1)
    graph_algo = GraphAlgo()
    graph_algo.g = graph_g
    # print(str(graph_algo.dijkstra(graph_g.get_all_v()[1], graph_g.get_all_v()[8])))
    print(str(graph_algo.shortest_path(1, 8)))

    # graph_scc_test = DiGraph()  # Test empty graph.
    # graph_scc_test = None  # Test None graph.
    graph_scc_test = DiGraph()
    for i in range(1, 6):
        graph_scc_test.add_node(i)
    graph_scc_test.add_edge(1, 2, 1)
    graph_scc_test.add_edge(2, 3, 2)
    graph_scc_test.add_edge(3, 1, 3)
    graph_scc_test.add_edge(3, 4, 4)
    graph_scc_test.add_edge(4, 5, 5)

    algo_scc_test = GraphAlgo(graph_scc_test)
    print(str(algo_scc_test.connected_components()))
    print(str(algo_scc_test.connected_component(4)))
