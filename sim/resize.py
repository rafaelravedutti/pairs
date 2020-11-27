from ast.branches import Filter
from ast.data_types import Type_Int
from ast.loops import While


class Resize:
    def __init__(self, sim, capacity_var, arrays, grow_fn=None):
        self.sim = sim
        self.capacity_var = capacity_var
        self.arrays = [arrays] if not isinstance(arrays, list) else arrays
        self.resize_var = self.sim.add_or_reuse_var('resize', Type_Int)
        self.grow_fn = grow_fn if grow_fn is not None else (lambda x: x * 2)

    def __iter__(self):
        self.resize_var.set(1)
        for _ in While(self.sim, self.resize_var > 0):
            self.resize_var.set(0)
            yield self.capacity_var, self.resize_var
            for _ in Filter(self.sim, self.resize_var > 0):
                self.capacity_var.set(self.grow_fn(self.resize_var))
                for a in self.arrays:
                    a.realloc()
