from typing import List
from queue import PriorityQueue
from collections import deque
import json
from src.GraphAlgoInterface import GraphAlgoInterface
from src.GraphInterface import GraphInterface
from src.DiGraph import DiGraph
from src.Node import Node
import numpy as np
import matplotlib.pyplot as plt


class GraphAlgo(GraphAlgoInterface):

    def __init__(self, graph=DiGraph()):
        self.g = graph

    def get_graph(self) -> GraphInterface:
        """
        :return: the directed graph on which the algorithm works on.
        """
        return self.g

    def load_from_json(self, file_name: str) -> bool:
        """
        Loads a graph from a json file.
        @param file_name: The path to the json file
        @returns True if the loading was successful, False o.w.
        """
        try:
            with open(file_name, "r") as file:  # open json file with read-only option.
                graph = DiGraph()
                json_obj = json.load(file)
                nodes = json_obj["Nodes"]
                edges = json_obj["Edges"]
                for node in nodes:
                    graph.add_node(node["id"], eval(str(node["pos"] if "pos" in node else None)))
                for edge in edges:
                    graph.add_edge(edge["src"], edge["dest"], edge["w"])
                self.g = graph
                return True
        except IOError as e:
            print(e)
            return False
        # graph = DiGraph()  # Create new empty graph object.
        # try:
        #     with open(file_name, "r") as file:  # open json file with read-only option.
        #         graph_dict = json.load(file)  # Load dictionary from file.
        #         v_dict = graph_dict["V"]  # A dictionary representing all nodes (vertices).
        #         e_dict = graph_dict["E"]  # A dictionary representing all edges.
        #         for v in v_dict.values():
        #             graph.add_node(v["key"], None if v["pos"] is None else tuple(v["pos"]))
        #         for k1, v1 in e_dict.items():
        #             inner_edges_dict = v1
        #             for k2, v2 in inner_edges_dict.items():
        #                 graph.add_edge(int(k1), int(k2), float(v2))
        #         graph.MC = graph_dict["MC"]
        #         self.g = graph
        #         return True
        # except IOError as e:
        #     print(e)
        #     return False

    def save_to_json(self, file_name: str) -> bool:
        """
        Saves the graph in JSON format to a file
        @param file_name: The path to the out file
        @return: True if the save was successful, False o.w.
        """
        try:
            with open(file_name, "w") as file:
                nodes = [{}] * self.g.v_size()
                edges = [{}] * self.g.e_size()
                index = 0
                for node_id in self.g.get_all_v().keys():
                    if self.g.get_all_v()[node_id].pos is None:
                        nodes[index] = {"id": node_id}
                    else:
                        nodes[index] = {"pos": "%.14f,%.14f,%.1f" % self.g.get_all_v()[node_id].pos, "id": node_id}
                    index += 1
                index = 0
                for src in self.g.E.keys():
                    for dest, w in self.g.E[src].items():
                        edges[index] = {"src": src, "dest": dest, "w": w}
                        index += 1
                json_obj = {"Edges": edges, "Nodes": nodes}
                json.dump(json_obj, fp=file)
                return True
        except IOError as e:
            print(e)
            return False
        # try:
        #     with open(file_name, "w") as file:
        #         json.dump(self.g, default=lambda o: o.as_dict(), indent=4, fp=file)
        #         return True
        # except IOError as e:
        #     print(e)
        #     return False

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        """
        Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm
        @param id1: The start node id
        @param id2: The end node id
        @return: The distance of the path, a list of the nodes ids that the path goes through

        Example:
#      >>> from GraphAlgo import GraphAlgo
#       >>> g_algo = GraphAlgo()
#        >>> g_algo.addNode(0)
#        >>> g_algo.addNode(1)
#        >>> g_algo.addNode(2)
#        >>> g_algo.addEdge(0,1,1)
#        >>> g_algo.addEdge(1,2,4)
#        >>> g_algo.shortestPath(0,1)
#        (1, [0, 1])
#        >>> g_algo.shortestPath(0,2)
#        (5, [0, 1, 2])

        Notes:
        If there is no path between id1 and id2, or one of them dose not exist the function returns (float('inf'),[])
        More info:
        https://en.wikipedia.org/wiki/Dijkstra's_algorithm
        """
        if (id1 not in self.g.get_all_v()) or (id2 not in self.g.get_all_v()):  # One of nodes or both doesn't exist.
            return tuple((float('inf'), []))
        if id1 == id2:  # The distance from node to itself is 0.
            return tuple((0, [id1]))
        dijkstra_pair = self.dijkstra(self.g.get_all_v()[id1], self.g.get_all_v()[id2])
        if dijkstra_pair is None:  # Means that there is no path
            return tuple((float('inf'), []))
        path = []  # Create empty path container.
        parent = id2  # Start from destination.
        while parent != id1:  # Go back from destination to src.
            path.insert(0, parent)  # Insert parent at index 0 of the path list.
            parent = dijkstra_pair[0][parent]  # Go to the next node index.
        path.insert(0, id1)  # Finally insert the src node id.
        return tuple((dijkstra_pair[1], path))  # Return shortest distance and path.

    def dijkstra(self, src: Node, dest: Node) -> (dict, float):
        parent_map = {}
        pq = PriorityQueue()
        for node in self.g.get_all_v().values():  # Initialize all nodes as unvisited.
            node.temp_weight = float('inf')
            node.temp_color = "WHITE"
        src.temp_weight = 0  # Set the src node weight to 0 (a node from itself)
        pq.put(src)  # Add src node to the priority queue
        while not pq.empty():  # while there are nodes to visit.
            current_node = pq.get()  # Remove the node with the minimum weight. O(Log(n)) operation.
            if current_node.temp_color == "WHITE":  # If node is unvisited.
                current_node.temp_color = "GRAY"  # Mark node as visited.
            else:
                continue  # If we already visited current_node, skip to the next node.
            if current_node.key == dest.key:  # If we found our destination node. then algorithm is done.
                return tuple((parent_map, dest.temp_weight))  # Return the a pair of (parent_map, minimum distance).
            if current_node.key in self.g.E:  # If current_node has neighbors at all.
                for neighbor in self.g.E[current_node.key].keys():  # For each neighbor of the current node.
                    if self.g.get_all_v()[neighbor].temp_color == "WHITE":  # If neighbor is unvisited.
                        path_dist = float(current_node.temp_weight) + float(self.g.E[current_node.key][neighbor])
                        if path_dist < self.g.get_all_v()[neighbor].temp_weight:  # If we found a shorter path.
                            self.g.get_all_v()[neighbor].temp_weight = path_dist  # Update the path.
                            parent_map[neighbor] = current_node.key  # Update parents map.
                            pq.put(self.g.get_all_v()[neighbor])  # Put neighbor node to the queue.
        return None  # If we get to this line. It means there is no such path.

    def connected_component(self, id1: int) -> list:
        """
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
        @param id1: The node id
        @return: The list of nodes in the SCC

        Notes:
        If the graph is None or id1 is not in the graph, the function should return an empty list []
        """
        if (self.g is None) or (self.g.v_size() == 0) or (id1 not in self.g.get_all_v()):
            return []
        component = self.scc_tarjan(id1)
        if component is None:
            return []
        else:
            return component

    def connected_components(self) -> List[list]:
        """
        Finds all the Strongly Connected Component(SCC) in the graph.
        @return: The list all SCC

        Notes:
        If the graph is None the function should return an empty list []
        """
        if (self.g is None) or (self.g.v_size() == 0):
            return []
        return self.scc_tarjan()

    def scc_tarjan(self, node_id=None):
        """
        Tarjan's - Strongly Connected Components (SCC) algorithm implementation.
        This algorithm is implemented according to the pseudo code in the next video:
        https://www.youtube.com/watch?v=wUgWX0nc4NY
        :return: a list of lists representing the strongly connected components.
        """
        components = []
        time = 0
        low_link = [0] * self.g.v_size()
        ids = [0] * self.g.v_size()
        on_stack = [False] * self.g.v_size()
        translator = {}  # Translates node ids to SCC node ids.
        index = 0
        stack = deque()
        for node in self.g.get_all_v().values():
            node.temp_color = "WHITE"  # Mark all nodes as unvisited
            translator[node.key] = index  # Initialize ids translator.
            index += 1
        if node_id is not None:
            node = self.g.get_all_v()[node_id]
            component = self.dfs(node, components, time, low_link, ids, on_stack, stack, translator, node_id)
            return component
        else:
            for node in self.g.get_all_v().values():
                if node.temp_color == "WHITE":  # If node is unvisited.
                    self.dfs(node, components, time, low_link, ids, on_stack, stack, translator, node_id)
            return components
        # for node in self.g.get_all_v().values():
        #     if node.temp_color == "WHITE":  # If node is unvisited.
        #         component = self.dfs(node, components, time, low_link, ids, on_stack, stack, translator, node_id)
        #         if component is not None:
        #             return component
        # if node_id is not None:
        #     return None
        # return components

    def dfs(self, node: Node, components, time, low_link, ids, on_stack, stack, translator, node_id=None):
        """
        DFS algorithm implementation used by Tarjan's algorithm.
        :param node: starting node.
        :param components: the list of strongly connected components which need to be updated.
        :param time: to give each node an id.
        :param low_link: the low link values (The smallest node ids reachable from current node).
        :param ids: a unique id list for each node.
        :param on_stack: tracks if whether or not nodes are on the stack.
        :param stack: the stack containing the nodes.
        :param translator: translates node id to index.
        :param node_id: optional: search for specific node_id SCC.
        :return: a list of SCCs or a single SCC if node_id is specified.
        """
        at = node.key  # Denote the id of the node that we are currently at.
        stack.append(at)  # Push current node to the stack.
        on_stack[translator[at]] = True  # Mark the current node as being on the stack.
        low_link[translator[at]] = ids[translator[at]] = time  # Give an id and a current low_link value to the node.
        time += 1
        node.temp_color = "GRAY"  # Mark current node as visited.
        if node.key in self.g.E:  # If current_node has neighbors at all.
            for neighbor in self.g.E[node.key].keys():  # For each neighbor of the current node.
                to = neighbor  # Represents the id of the node that we are going to.
                if self.g.get_all_v()[to].temp_color == "WHITE":  # If the neighbor is unvisited.
                    self.dfs(self.g.get_all_v()[to], components, time, low_link, ids, on_stack, stack, translator,
                             node_id)
                if on_stack[translator[to]]:  # If the node that we are just came from is on the stack.
                    low_link[translator[at]] = min(low_link[translator[at]], low_link[translator[to]])
        # After visiting all neighbors of the current node.
        if ids[translator[at]] == low_link[translator[at]]:  # Check if we are on the beginning of a SCC.
            component = []  # Build a list which will hold the component.
            while True:
                x = stack.pop()  # Pop all the nodes inside the strongly connected component.
                component.insert(0, x)  # Add this node to the beginning of the list of SCCs.
                on_stack[translator[x]] = False  # Mark node as no longer being on stack.
                low_link[translator[x]] = ids[translator[at]]  # Make sure all nodes on the same SCC have the same id.
                if x == at:  # Once we reach the start of the SCC, break the loop.
                    break
            if node_id is not None:  # If SCC node_id is specified.
                if node_id in component:  # If we found the desired SCC.
                    return component  # Return the component containing node_id.
            components.append(component)  # Finally add the SCC to the list of SCCs.

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        plt.figure(figsize=(10, 8))
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Directed Graph: " + str(self.g.v_size()) + " Nodes, " + str(self.g.e_size()) + " Edges")
        positions = {}
        for node in self.g.get_all_v().values():  # Draw nodes
            if node.pos is not None:
                x_pos = node.pos[0]
                y_pos = node.pos[1]
                positions[node.key] = (x_pos, y_pos)
                plt.scatter(x=x_pos, y=y_pos, s=80, zorder=2, lw=3, c='#0000ff')
            else:
                a1 = 35
                a2 = 32
                rand_x = (np.random.rand()+a1)
                rand_y = (np.random.rand()+a2)
                positions[node.key] = (rand_x, rand_y)
                plt.scatter(x=rand_x, y=rand_y, s=80, zorder=2, lw=3, c='#0000ff')
        for src in self.g.E.keys():
            for dest, w in self.g.E[src].items():
                plt.plot([positions[src][0], positions[dest][0]],
                         [positions[src][1], positions[dest][1]],
                         'C3', zorder=1, lw=3, c='#000000')
        plt.show()

        # x_values = [35.18753053591606, 35.18958953510896, 35.19341035835351, 35.197528356739305]
        # y_values = [32.10378225882353, 32.10785303529412, 32.10610841680672, 32.1053088]
        # plt.xlabel("X axis")
        # plt.ylabel("Y axis")
        # plt.title("Directed Graph: " + str(self.g.v_size()) + " Nodes, " + str(self.g.e_size()) + " Edges")
        # # plt.plot([1, 2], [4, 5], 'C3', zorder=1, lw=3, c='#000000')
        # plt.scatter(x_values, y_values, s=100, zorder=2, c='#0000ff')
        # # plt.text(2, 5, 10, fontsize=15)
        # # plt.annotate(10, (1, 4))
        # plt.show()
