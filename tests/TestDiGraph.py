import unittest
from src.DiGraph import DiGraph


class MyTestCase(unittest.TestCase):
    """ Basic tests:
        1) Test adding one node.
        2) Test adding two nodes with an edge.
        3) Test empty graph
    """

    """ 1 """
    def test_add_node(self):
        graph = DiGraph()
        graph.add_node(0)
        self.assertIn(0, graph.V)

    """ 2 """
    def test_add_edge(self):
        graph = DiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_edge(1, 2, 0.4)
        self.assertEqual(0.4, graph.E[1][2])

    """ 3 """
    def test_empty_graph(self):
        graph = DiGraph()
        self.assertEqual(0, graph.v_size())
        self.assertEqual(0, graph.e_size())
        self.assertEqual(0, graph.get_mc())

    """ 4 """
    def test_v_size(self):
        graph = DiGraph()
        for i in range(10):
            graph.add_node(i)
        self.assertEqual(10, graph.node_size)

    """ 5 """
    def test_e_size(self):
        graph = DiGraph()
        for i in range(10):
            graph.add_node(i)
        for i in range(1, 10):
            graph.add_edge(i-1, i, 1.0)
        self.assertEqual(9, graph.edge_size)

    """ 6 """
    def test_remove_edge(self):
        graph = DiGraph()
        for i in range(10):
            graph.add_node(i)
        for i in range(1, 10):
            graph.add_edge(i-1, i, 1.0)
        self.assertEqual(9, graph.edge_size)
        self.assertEqual(1.0, graph.E[4][5])
        graph.remove_edge(4, 5)
        self.assertEqual(8, graph.edge_size)

    """ 7 """
    def test_remove_node(self):
        graph = DiGraph()
        for i in range(10):
            graph.add_node(i)
        for i in range(1, 10):
            graph.add_edge(i-1, i, 1.0)
        graph.remove_node(9)
        self.assertEqual(9, graph.node_size)
        self.assertEqual(8, graph.edge_size)

    """ 8 """
    def test_all_in_edges_of_node(self):
        graph = DiGraph()
        for i in range(10):
            graph.add_node(i)
        graph.add_edge(1, 2, 1.2)
        graph.add_edge(1, 3, 1.2)
        graph.add_edge(1, 4, 1.2)
        graph.add_edge(1, 5, 1.2)
        graph.add_edge(1, 5, 1.2)

        graph.add_edge(9, 1, 14.0)
        graph.add_edge(9, 1, 14.0)
        graph.add_edge(8, 1, 15.0)
        self.assertEqual(2, len(graph.all_in_edges_of_node(1)))

    """ 9 """
    def test_all_out_edges_of_nodes(self):
        graph = DiGraph()
        for i in range(10):
            graph.add_node(i)
        graph.add_edge(1, 2, 1.2)
        graph.add_edge(1, 3, 1.2)
        graph.add_edge(1, 4, 1.2)
        graph.add_edge(1, 5, 1.2)
        graph.add_edge(1, 5, 1.2)

        graph.add_edge(9, 1, 14.0)
        graph.add_edge(9, 1, 14.0)
        graph.add_edge(8, 1, 15.0)
        self.assertEqual(4, len(graph.all_out_edges_of_node(1)))

    """ 10 """
    def test_mc(self):
        graph = DiGraph()
        for i in range(10):
            graph.add_node(i)
        self.assertEqual(10, graph.get_mc())
        graph.add_node(9)  # Add existing node. should not affect graph size.
        self.assertEqual(10, graph.get_mc())
        graph.add_edge(1, 2, 1.2)
        graph.add_edge(1, 3, 1.2)
        graph.add_edge(1, 4, 1.2)
        graph.add_edge(1, 5, 1.2)
        self.assertEqual(14, graph.get_mc())
        graph.add_edge(1, 5, 5)  # Should not affect MC (edge already exist).
        self.assertEqual(14, graph.get_mc())
        graph.remove_edge(1, 8)  # Should not affect MC.
        self.assertEqual(14, graph.get_mc())
        graph.remove_edge(1, 5)  # Should increase MC
        self.assertEqual(15, graph.get_mc())
        graph.remove_node(1)  # Should increase MC by 4
        self.assertEqual(19, graph.get_mc())


if __name__ == '__main__':
    unittest.main()
