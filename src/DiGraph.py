from src.GraphInterface import GraphInterface
from src.Node import Node


class DiGraph(GraphInterface):
    """
    This class represents a directed (positive) weighted graph data structure G(V, E)
    """
    def __init__(self):
        """
        Default constructor:
        node_size - current number of nodes in this graph.
        edge_size - current number of edges in this graph.
        MC - current number of modifications in this graph.
        V - a dictionary of all nodes (vertices) in this graph.
        E - a dictionary of all edges in this graph.
        R - a dictionary of the reverse form of all edges in this graph.
        """
        self.node_size = 0
        self.edge_size = 0
        self.MC = 0
        self.V = {}
        self.E = {}
        self.R = {}

    def deep_copy(self) -> GraphInterface:
        """
        Return a deep copy of this graph.
        @return: a graph object representing a deep copy of this graph.
        """
        graph = DiGraph()
        for node in self.V.keys():
            graph.add_node(node)
        for k1, v1 in self.E.items():
            inner_nodes_dict = v1
            for k2, v2 in inner_nodes_dict.items():
                graph.add_edge(k1, k2, v2)
        graph.node_size = self.node_size
        graph.edge_size = self.edge_size
        graph.MC = self.MC
        return graph

    def v_size(self) -> int:
        """
        Returns the number of vertices in this graph
        @return: The number of vertices in this graph
        """
        return self.node_size

    def e_size(self) -> int:
        """
        Returns the number of edges in this graph
        @return: The number of edges in this graph
        """
        return self.edge_size

    def get_all_v(self) -> dict:
        """return a dictionary of all the nodes in the Graph, each node is represented using a pair
         (node_id, node_data)
        """
        return self.V

    def all_in_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected to (into) node_id ,
        each node is represented using a pair (other_node_id, weight)
         """
        if id1 not in self.R:  # If there is no in edges, return empty dictionary.
            return {}
        return self.R[id1]

    def all_out_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected from node_id , each node is represented using a pair
        (other_node_id, weight)
        """
        if id1 not in self.E:  # If there is no out edges, return empty dictionary.
            return {}
        return self.E[id1]

    def get_mc(self) -> int:
        """
        Returns the current version of this graph,
        on every change in the graph state - the MC should be increased
        @return: The current version of this graph.
        """
        return self.MC

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        """
        Adds an edge to the graph.
        @param id1: The start node of the edge
        @param id2: The end node of the edge
        @param weight: The weight of the edge
        @return: True if the edge was added successfully, False o.w.

        Note: If the edge already exists or one of the nodes dose not exists the functions will do nothing
        """
        if (id1 not in self.V) or (id2 not in self.V) or (id1 == id2) or (weight < 0):
            return False
        if id1 in self.E:
            if id2 in self.E[id1]:  # Check if the edge already exist.
                return False
            else:  # If inner key doesn't exist, then add (id2, weight)
                self.E[id1][id2] = weight
                self.add_reverse_edge(id1, id2, weight)  # Add the reverse edge as well
                self.edge_size += 1
                self.MC += 1
                return True
        else:  # If id1 is not in E, Add new dictionary at E[id1]
            self.E[id1] = {id2: weight}  # Add an edge between id1 to id2 with weight
            self.add_reverse_edge(id1, id2, weight)  # Add the reverse edge as well
            self.edge_size += 1
            self.MC += 1
            return True

    def add_reverse_edge(self, id1: int, id2: int, weight: float):
        """
        Helper functions. Saves the reverse edge of the given edge in R dictionary.
        @param id1: Original edge source node key.
        @param id2: Original edge destination node key.
        @param weight: Original edge weight.
        @return: None
        """
        if id2 in self.R:
            self.R[id2][id1] = weight  # Update current dictionary
        else:
            self.R[id2] = {id1: weight}  # Create new dictionary

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
        Adds a node to the graph.
        @param node_id: The node ID
        @param pos: The position of the node
        @return: True if the node was added successfully, False o.w.

        Note: if the node id already exists the node will not be added
        """
        if (node_id < 0) or (node_id in self.V):  # If key id is negative or already exists. do nothing.
            return False
        self.V[node_id] = Node(node_id, pos)  # Add a new vertex
        self.node_size += 1  # Increment node size by 1
        self.MC += 1
        return True

    def remove_node(self, node_id: int) -> bool:
        """
        Removes a node from the graph.
        @param node_id: The node ID
        @return: True if the node was removed successfully, False o.w.

        Note: if the node id does not exists the function will do nothing
        """
        if node_id not in self.V:  # Check if node_id is not present in V. If so, then do nothing.
            return False
        for i in list(self.all_out_edges_of_node(node_id)):  # Remove all out edges
            self.remove_edge(node_id, i)
            self.MC -= 1
        for i in list(self.all_in_edges_of_node(node_id)):  # Remove all in edges
            self.remove_edge(i, node_id)
            self.MC -= 1
        self.V.pop(node_id)  # Finally remove the node itself
        if node_id in self.E:
            self.E.pop(node_id)  # Remove node from edges list
        if node_id in self.R:
            self.R.pop(node_id)  # Remove node from the dummy reverse edges list
        self.node_size -= 1
        self.MC += 1
        return True

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        """
        Removes an edge from the graph.
        @param node_id1: The start node of the edge
        @param node_id2: The end node of the edge
        @return: True if the edge was removed successfully, False o.w.

        Note: If such an edge does not exists the function will do nothing
        """
        if (node_id1 not in self.V) or (node_id2 not in self.V) or (node_id1 == node_id2):  # Check if nodes exist
            return False
        if node_id1 not in self.E:  # Check if node_id1 has edges
            return False
        if node_id2 not in self.E[node_id1]:  # Check if there is an edge between node_id1 to node_id2
            return False
        self.E[node_id1].pop(node_id2)  # Remove the edge (node_id1 -> node_id2)
        self.R[node_id2].pop(node_id1)  # Remove the dummy reverse edge (node_id2 -> node_id1)
        self.edge_size -= 1
        self.MC += 1
        return True

    def as_dict(self):
        """
        Returns a deep copy of this graph as a dictionary.
        @return: Dictionary representation of this graph.
        """
        self_copy_dict = self.deep_copy().__dict__
        del self_copy_dict["R"]
        return self_copy_dict

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            a = self.V == other.V
            b = self.E == other.E
            return a and b
        else:
            return False

    def __str__(self):
        """
        Returns a string representation of this graph.
        @return: String representing this graph.
        """
        to_return = ""
        for i in self.V:
            to_return += str(i) + " -> " + str(self.E[i] if i in self.E else "{no edges}") + "\n"
        return to_return
