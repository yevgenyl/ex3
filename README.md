# Directed Weighted Graph Data Structure and Algorithms
## Python implementation
### Project Overview

- This project was built as part of the OOP Course of the Computer Science Department at Ariel University.

- This project describes a directed weighted graph data structure and Graph Theory algrithms.

- For more information about directed graphs please read: 
https://en.wikipedia.org/wiki/Directed_graph

- As part of this project, we are asked to compare running time differneces between this implementation
to the well known python's open source library **NetworkX**. for more inforamtion about NetworkX please visit: https://github.com/networkx/networkx

- Inside the src folder you can find an implementation of the Graph Theory data structure and algorithms.

- For more inforamtion, including the comparison results with NetworkX please read the **Wiki page** of this project.

- **Project contributors**: Evgueni Lachtchenov

- Below is an illustration of a directed Weighted Graph **G(V,E)** with 40 vertices and 102 directed edges:

![Image of undirected weighted graph](https://github.com/yevgenyl/ex1/blob/master/res/graph_image.png?raw=true)

### Class and Interface Summary

## DiGraph Class

### Class Overview
- This class represents a directed (positive) weighted graph data structure G(V, E)

- It supports a large number of nodes (over 10^6, with average degree of 10).

- The implementation is based on a compact and efficient representation using python dictionaries.

- Most of the basic operations (add node, connect nodes.. etc) are running in a constant time O(1).

## GraphAlgo

### Class Overview
- This class represents a directed (positive) Weighted Graph Theory algorithms including:

  - shortest_path
  - Strongly Connected Components (SCC)
  - Strongly Connected Component which includes a specific vertex.
  - Save to Json
  - Load from Json

### Algorithms
  
  - **SCC** - The code conatains two implementations:
    1) Tarjan's SCC algorithm - the recursive version usind DFS algorithm.
    2) Simple iterative algorithm which searches for intersections.
  - **BFS - Breadth-first search:** This algorithm is used by the SCC algorithm.
  - **Dijkstra:** This algorithm is used by `shortest_path` method to find the shortest path between two vertices.

## Unit Tests
- This project was tested using python unit tests.
- Inside the tests folder you can find two test classes:
  - **TestDiGraph:** this class was used to test the DiGraph class.
  - **TestGraphAlgo:** this class was used to test the GraphAlgo class. 

## Importing and Using the Project
- In order to be able to use this project, you should have the following libraries installed:

  - cycler==0.10.0
  - kiwisolver==1.3.1
  - matplotlib==3.3.3
  - numpy==1.19.5
  - Pillow==8.1.0
  - pyparsing==2.4.7
  - python-dateutil==2.8.1
  - six==1.15.0
  - networkx~=2.5

- Simply clone this project to you computer and then import it to your favorite IDE (PyCharm, etc..).

- Then you should create a new class with `main` method or use it in any other way that you prefer.

- Below is an example of creating a new directed weighted graph and adding a vertices to it:
  ```python
  g = DiGraph()  # Creates an empty directed graph
  for n in range(4):  # Adding 4 nodes to the graph with ids (0 to 3)
    g.add_node(n)
  g.add_edge(0, 1, 1). # Add an edge between node 0 to node 1 with weight of 1.
  ``` 
