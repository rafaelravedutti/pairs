from pairs.ir.assign import Assign
from pairs.ir.block import pairs_device_block, pairs_inline
from pairs.ir.loops import ParticleFor
from pairs.ir.memory import Malloc, Realloc
from pairs.ir.properties import RegisterProperty, RegisterContactProperty
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.lowerable import Lowerable, FinalLowerable
from functools import reduce
import operator


class AllocateProperties(FinalLowerable):
    def __init__(self, sim, realloc=False):
        self.sim = sim
        self.realloc = realloc

    @pairs_inline
    def lower(self):
        for p in self.sim.properties.all():
            sizes = []
            if Types.is_real(p.type()) or Types.is_integer(p.type()):
                sizes = [self.sim.particle_capacity]
            elif p.type() == Types.Vector:
                sizes = [self.sim.particle_capacity, self.sim.ndims()]
            elif p.type() == Types.Matrix:
                sizes = [self.sim.particle_capacity, self.sim.ndims() * self.sim.ndims()]
            elif p.type() == Types.Quaternion:
                sizes = [self.sim.particle_capacity, self.sim.ndims() + 1]
            else:
                raise Exception("Invalid property type!")

            if self.realloc:
                UpdateProperty(self.sim, p, sizes)
            else:
                RegisterProperty(self.sim, p, sizes)


class AllocateContactProperties(FinalLowerable):
    def __init__(self, sim, realloc=False):
        self.sim = sim

    @pairs_inline
    def lower(self):
        for p in self.sim.contact_properties:
            sizes = []
            if Types.is_real(p.type()) or Types.is_integer(p.type()):
                sizes = [self.sim.particle_capacity * self.sim.neighbor_capacity]
            elif p.type() == Types.Vector:
                sizes = [self.sim.particle_capacity * self.sim.neighbor_capacity, self.sim.ndims()]
            elif p.type() == Types.Matrix:
                sizes = [self.sim.particle_capacity * self.sim.neighbor_capacity, self.sim.ndims() * self.sim.ndims()]
            elif p.type() == Types.Quaternion:
                sizes = [self.sim.particle_capacity * self.sim.neighbor_capacity, self.sim.ndims() + 1]
            else:
                raise Exception("Invalid contact property type!")

            RegisterContactProperty(self.sim, p, sizes)


class ResetVolatileProperties(Lowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_device_block
    def lower(self):
        self.sim.module_name("reset_volatile_properties")
        for i in ParticleFor(self.sim):
            for p in self.sim.properties.volatiles():
                Assign(self.sim, p[i], 0.0)
