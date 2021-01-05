class Print:
    def __init__(self, sim, string):
        self.sim = sim
        self.string = string

    def __str__(self):
        return f"Print<{self.string}>"

    def children(self):
        return []

    def transform(self, fn):
        return fn(self)
