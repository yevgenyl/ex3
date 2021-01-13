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

    """
    This class represents a set of algorithms applicable on a directed weighted graph including:
    1) Shortest Path - Dijkstra
    2) Strongly connected Components (SCC) -
        1) Tarjan's recursive algorithm
        2) Simple iterative algorithm using BFS (searches for intersections).
    3) Strongly connected component - using BFS
    4) Save to Json
    5) Load from Json
    """

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
        if (self.g is None) or (id1 not in self.g.get_all_v()) or \
                (id2 not in self.g.get_all_v()):  # One of nodes or both doesn't exist.
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
        return self.scc(id1)

    def connected_components(self) -> List[list]:
        """
        Finds all the Strongly Connected Component(SCC) in the graph.
        @return: The list all SCC

        Notes:
        If the graph is None the function should return an empty list []
        """
        if (self.g is None) or (self.g.v_size() == 0):
            return []
        return self.scc()

    def scc(self, node_id=None):
        """
        Strongly connected components (SCC)
        This functions used both for SCCs and for a single SCC (by id)
        :param node_id: optional - If id is not node, then search for a specific component.
        :return: the Strongly Connected Component(SCC) that node id1 is a part of. or all SCCs.
        """
        components = []
        temp_dict = {}
        if node_id is None:
            for n in self.g.get_all_v().values():
                if n.key not in temp_dict:
                    list_out = self.bfs(n, True)
                    list_in = self.bfs(n, False)
                    intersect = list(set(list_out).intersection(list_in))
                    components.append(intersect)
                    temp = dict.fromkeys(intersect, 1)
                    temp_dict.update(temp)
            return components
        else:
            n = self.g.get_all_v()[node_id]
            list_out = self.bfs(n, True)
            list_in = self.bfs(n, False)
            intersect = list(set(list_out).intersection(list_in))
            return intersect

    def bfs(self, node: Node, out: bool) -> list:
        """
        BFS algorithm
        Helper algorithm for the SCC algorithm.
        :param node: - a node to start BFS searching from.
        :param out: - determines how to search: out - search all out edges of node, in - search al in edges of node.
        :return: - a list representing all visited nodes.
        """
        for n in self.g.get_all_v().values():  # Set all nodes as unvisited.
            n.temp_color = "WHITE"

        visited = []
        queue = []

        node.temp_color = "GRAY"
        visited.append(node.key)
        queue.append(node.key)

        if out:
            while queue:
                v = queue.pop(0)
                for neighbor in self.g.all_out_edges_of_node(v).keys():
                    if self.g.get_all_v()[neighbor].temp_color == "WHITE":
                        self.g.get_all_v()[neighbor].temp_color = "GRAY"
                        visited.append(neighbor)
                        queue.append(neighbor)
        else:
            while queue:
                v = queue.pop(0)
                for neighbor in self.g.all_in_edges_of_node(v).keys():
                    if self.g.get_all_v()[neighbor].temp_color == "WHITE":
                        self.g.get_all_v()[neighbor].temp_color = "GRAY"
                        visited.append(neighbor)
                        queue.append(neighbor)
        return visited

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
        plt.figure(figsize=(11, 8))
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        if (self.g is None) or (self.g.v_size() == 0):  # If it's an empty graph.
            plt.title("Directed Graph: " + str(0) + " Nodes, " + str(0) + " Edges")
            plt.text(0.35, 0.5, "Empty Graph", fontsize=24, zorder=3, color='#DD2BDA', weight='bold')
            plt.show()
        else:  # If graph is not empty
            plt.title("Directed Graph: " + str(self.g.v_size()) + " Nodes, " + str(self.g.e_size()) + " Edges")
            positions = {}  # Create container for storing nodes positions.
            for node in self.g.get_all_v().values():  # Draw nodes
                if node.pos is not None:
                    x_pos = node.pos[0]
                    y_pos = node.pos[1]
                    positions[node.key] = (x_pos, y_pos)
                    plt.scatter(x=x_pos, y=y_pos, s=30, zorder=2, lw=3, c='#0000ff')
                    plt.text(x_pos, y_pos+0.0002, node.key, fontsize=12, zorder=3, color='#DD2BDA', weight='bold')
                else:
                    a1 = 35.0
                    a2 = 32.0
                    rand_x = (np.random.ranf()+a1)
                    rand_y = (np.random.ranf()+a2)
                    positions[node.key] = (rand_x, rand_y)
                    plt.scatter(x=rand_x, y=rand_y, s=30, zorder=2, lw=3, c='#0000ff')
                    plt.text(rand_x, rand_y+0.01, node.key, fontsize=12, zorder=3, color='#DD2BDA', weight='bold')
            for src in self.g.E.keys():
                for dest, w in self.g.E[src].items():
                    # plt.plot([positions[src][0], positions[dest][0]],
                    #          [positions[src][1], positions[dest][1]],
                    #          'C3', zorder=1, lw=3, c='#000000')
                    x1 = positions[src][0]
                    y1 = positions[src][1]
                    x2 = positions[dest][0]
                    y2 = positions[dest][1]
                    dx = x2 - x1
                    dy = y2 - y1
                    plt.arrow(x1, y1, dx, dy, head_width=0.0002, head_length=0.0003,
                              fc='lightblue', ec='black', length_includes_head=True, width=0.00001)
            plt.show()
