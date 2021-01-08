class AlgoHelper:

    def __init__(self):
        self.data = {}

    def store_temporal_weight(self, key, value: float):
        if key not in self.data:
            self.data[key] = [value, ""]
        else:
            self.data[key][0] = value

    def store_temporal_color(self, key, color: str):
        if key not in self.data:
            self.data[key] = [float('inf'), str]
        else:
            self.data[key][1] = str

    def get_weight(self, key) -> float:
        if key not in self.data:
            return float('inf')
        else:
            return self.data[key][0]

    def get_color(self, key) -> str:
        if key not in self.data:
            return "WHITE"
        else:
            return self.data[key][1]
