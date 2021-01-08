import math


class Node:

    def __init__(self, node_id: int, pos: tuple):
        self.key = node_id
        self.pos = pos
        self.temp_weight = float('inf')
        self.temp_color = "WHITE"

    def as_dict(self):
        self_dict = self.__dict__
        return self_dict

    def __lt__(self, other):
        return self.temp_weight < other.temp_weight

    def __eq__(self, other):
        math.isclose(self.temp_weight, other.temp_weight)

    def __gt__(self, other):
        return self.temp_weight > other.temp_weight

    def __repr__(self):
        return str(self.key)
