from pairs.ir.branches import Filter
from pairs.ir.data_types import Type_Int, Type_Float, Type_Vector
from pairs.ir.loops import While
from pairs.ir.memory import Realloc
from pairs.ir.properties import UpdateProperty
from pairs.ir.utils import Print
from functools import reduce
import operator


class Resize:
    def __init__(self, sim, capacity_var, grow_fn=None):
        self.sim = sim
        self.capacity_var = capacity_var
        self.resize_var = self.sim.add_or_reuse_var('resize', Type_Int)
        self.grow_fn = grow_fn if grow_fn is not None else (lambda x: x * 2)

    def __iter__(self):
        properties = self.sim.properties
        self.resize_var.set(1)
        for _ in While(self.sim, self.resize_var > 0):
            self.resize_var.set(0)
            yield self.resize_var
            for _ in Filter(self.sim, self.resize_var > 0):
                self.sim.add_statement(Print(self.sim, f"Resize {self.capacity_var.name()}"))
                self.capacity_var.set(self.grow_fn(self.resize_var))
                for a in self.capacity_var.bonded_arrays():
                    a.realloc()

                if properties.is_capacity(self.capacity_var):
                    capacity = sum(self.sim.properties.capacities)
                    for p in properties.all():
                        if p.type() == Type_Vector:
                            sizes = [capacity, self.sim.ndims()]
                        else:
                            sizes = [capacity]

                        Realloc(self.sim, p, reduce(operator.mul, sizes))
                        UpdateProperty(self.sim, p, sizes)