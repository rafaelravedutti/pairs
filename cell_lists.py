from assign import AssignAST
from block import BlockAST
from branches import BranchAST
from cast import CastAST
from data_types import Type_Int
from functools import reduce
from expr import ExprAST
from loops import ForAST, ParticleForAST
import math

class CellLists:
    def __init__(self, sim, spacing, cutoff_radius):
        nneighbor_cells = [math.ceil(cutoff_radius / (spacing if not isinstance(spacing, list) else spacing[d])) for d in range(0, sim.dimensions)]
        nstencil = reduce((lambda x, y: x * y), [nneighbor_cells[d] * 2 + 1 for d in range(0, sim.dimensions)])
        self.sim = sim
        self.spacing = spacing
        self.ncells = self.sim.add_array('ncells', self.sim.dimensions, Type_Int)
        self.ncells_total = self.sim.add_var('ncells_total', Type_Int)
        self.cell_capacity = self.sim.add_var('cell_capacity', Type_Int)
        self.cell_particles = self.sim.add_array('cell_particles', self.ncells_total * self.cell_capacity, Type_Int)
        self.cell_sizes = self.sim.add_array('cell_sizes', self.ncells_total, Type_Int)
        self.stencil = self.sim.add_array('stencil', nstencil, Type_Int)

    def build(self):
        positions = self.sim.property('position')
        reset_loop = ForAST(self.sim, 0, self.ncells_total)
        reset_loop.set_body(BlockAST([self.cell_sizes[reset_loop.iter()].set(0)]))

        fill_loop = ParticleForAST(self.sim)
        cell_index = [CastAST.int((positions[fill_loop.iter()][d] - self.sim.grid_config[d][0]) / self.spacing) for d in range(0, self.sim.dimensions)]
        flat_index = None
        for d in range(0, self.sim.dimensions):
            flat_index = cell_index[d] if flat_index is None else flat_index * self.ncells[d] + cell_index[d]

        fill_loop.set_body(BlockAST([
            BranchAST.if_stmt(ExprAST.and_op(flat_index >= 0, flat_index <= self.ncells_total), [
                self.cell_particles[flat_index * self.cell_capacity + self.cell_sizes[flat_index]].set(fill_loop.iter()),
                self.cell_sizes[flat_index].set(self.cell_sizes[flat_index] + 1)
            ])
        ]))

        return BlockAST([reset_loop, fill_loop])
