import unittest
from src.DiGraph import DiGraph
from src.GraphAlgo import GraphAlgo


class MyTestCase(unittest.TestCase):

    def test_save_load(self):
        # _____________ Test Load _____________
        algo1 = GraphAlgo()
        algo1.load_from_json("../data/A5")
        algo2 = GraphAlgo()
        algo2.load_from_json("../data/A5")
        self.assertEqual(algo1.get_graph(), algo2.get_graph())  # Should be equal
        algo2.get_graph().remove_node(40)
        self.assertNotEqual(algo1.get_graph(), algo2.get_graph())  # Should no be equal (one node is removed).
        # _____________ Test Save _____________
        algo3 = GraphAlgo()
        algo3.load_from_json("../data/A5")
        algo3.save_to_json("graph.json")
        algo4 = GraphAlgo()
        algo4.load_from_json("graph.json")
        self.assertEqual(algo3.get_graph(), algo4.get_graph())

    def test_shortest_path(self):
        graph = DiGraph()
        for i in range(1, 10):
            graph.add_node(i)
        graph.add_edge(1, 2, 1)
        graph.add_edge(1, 4, 30)
        graph.add_edge(1, 5, 400)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)
        graph.add_edge(5, 4, 1)
        graph.add_edge(4, 6, 1)
        graph.add_edge(5, 7, 1)
        graph.add_edge(3, 6, 50)
        graph.add_edge(7, 6, 1)
        graph.add_edge(6, 9, 1)
        graph.add_edge(6, 8, 1)
        graph_algo = GraphAlgo(graph)
        expected = (5.0, [1, 2, 3, 4, 6, 8])
        self.assertEqual(expected, graph_algo.shortest_path(1, 8))
        expected = (float('inf'), [])
        self.assertEqual(expected, graph_algo.shortest_path(9, 1))
        graph.remove_edge(1, 5)
        graph.add_edge(1, 5, 1)
        graph.remove_edge(5, 4)
        graph.add_edge(5, 4, 400)
        expected = (4.0, [1, 5, 7, 6, 8])
        self.assertEqual(expected, graph_algo.shortest_path(1, 8))
        self.assertEqual((0, [3]), graph_algo.shortest_path(3, 3))
        graph_algo = GraphAlgo(None)
        expected = (float('inf'), [])
        self.assertEqual(expected, graph_algo.shortest_path(9, 1))
        g_algo = GraphAlgo()
        graph = g_algo.get_graph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        graph.add_node(4)
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(1, 4, 0.5)
        graph.add_edge(2, 3, 0.3)
        graph.add_edge(4, 3, 0.1)
        self.assertEqual((0.6, [1, 4, 3]), g_algo.shortest_path(1, 3))

    def test_connected_components(self):
        graph = DiGraph()
        for i in range(1, 9):
            graph.add_node(i)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)
        graph.add_edge(4, 1, 1)

        graph.add_edge(3, 5, 1)
        graph.add_edge(5, 6, 1)
        graph.add_edge(6, 7, 1)
        graph.add_edge(7, 5, 1)

        graph.add_edge(7, 8, 1)

        algo = GraphAlgo(graph)

        self.assertEqual([[1, 2, 3, 4], [5, 6, 7], [8]], algo.connected_components())
        graph.add_edge(5, 3, 1)
        self.assertEqual([[1, 2, 3, 4, 5, 6, 7], [8]], algo.connected_components())
        algo = GraphAlgo()
        self.assertEqual([], algo.connected_components())
        algo = GraphAlgo(None)
        self.assertEqual([], algo.connected_components())

    def test_connected_component(self):
        graph = DiGraph()
        for i in range(1, 9):
            graph.add_node(i)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)
        graph.add_edge(4, 1, 1)

        graph.add_edge(3, 5, 1)
        graph.add_edge(5, 6, 1)
        graph.add_edge(6, 7, 1)
        graph.add_edge(7, 5, 1)

        graph.add_edge(7, 8, 1)

        algo = GraphAlgo(graph)
        self.assertEqual([1, 2, 3, 4], algo.connected_component(1))
        self.assertEqual([5, 6, 7], algo.connected_component(6))
        self.assertEqual([8], algo.connected_component(8))
        self.assertEqual([], algo.connected_component(10))
        algo = GraphAlgo()
        self.assertEqual([], algo.connected_component(10))
        algo = GraphAlgo(None)
        self.assertEqual([], algo.connected_component(10))

    def test_plot_graph(self):
        # _____________ Test plot empty graph  _____________
        algo = GraphAlgo()
        algo.plot_graph()
        # _____________ Test plot with random positions _____________
        graph = DiGraph()
        for i in range(1, 9):
            graph.add_node(i)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)
        graph.add_edge(4, 1, 1)

        graph.add_edge(3, 5, 1)
        graph.add_edge(5, 6, 1)
        graph.add_edge(6, 7, 1)
        graph.add_edge(7, 5, 1)

        graph.add_edge(7, 8, 1)

        algo = GraphAlgo(graph)
        algo.plot_graph()
        # _____________ Test plot with constant positions _____________
        graph = DiGraph()
        for i in range(1, 9):
            graph.add_node(i, (i, i+1, 0))
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)
        graph.add_edge(4, 1, 1)

        graph.add_edge(3, 5, 1)
        graph.add_edge(5, 6, 1)
        graph.add_edge(6, 7, 1)
        graph.add_edge(7, 5, 1)

        graph.add_edge(7, 8, 1)

        algo = GraphAlgo(graph)
        algo.plot_graph()
        # _____________ Test plot from A5 graph _____________
        algo = GraphAlgo()
        algo.load_from_json("../data/A5")
        algo.plot_graph()
        # _____________ Test plot from A4 graph _____________
        algo = GraphAlgo()
        algo.load_from_json("../data/A4")
        algo.plot_graph()
        # _____________ Test plot from A3 graph _____________
        algo = GraphAlgo()
        algo.load_from_json("../data/A3")
        algo.plot_graph()
        # _____________ Test plot from A2 graph _____________
        algo = GraphAlgo()
        algo.load_from_json("../data/A2")
        algo.plot_graph()
        # _____________ Test plot from A1 graph _____________
        algo = GraphAlgo()
        algo.load_from_json("../data/A1")
        algo.plot_graph()
        # _____________ Test plot from A0 graph _____________
        algo = GraphAlgo()
        algo.load_from_json("../data/A0")
        algo.plot_graph()


if __name__ == '__main__':
    unittest.main()
