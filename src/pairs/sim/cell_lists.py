from functools import reduce
import math
from pairs.ir.bin_op import BinOp
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
from pairs.ir.data_types import Type_Int
from pairs.ir.math import Ceil
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.sim.resize import Resize


class CellLists:
    def __init__(self, sim, grid, spacing, cutoff_radius):
        self.sim = sim
        self.grid = grid
        self.spacing = spacing if isinstance(spacing, list) else [spacing for d in range(sim.ndims())]
        self.cutoff_radius = cutoff_radius
        self.nneighbor_cells = [math.ceil(cutoff_radius / self.spacing[d]) for d in range(sim.ndims())]
        self.nstencil = self.sim.add_var('nstencil', Type_Int)
        self.nstencil_max = reduce((lambda x, y: x * y), [self.nneighbor_cells[d] * 2 + 1 for d in range(sim.ndims())])
        self.ncells = self.sim.add_var('ncells', Type_Int, 1)
        self.ncells_capacity = self.sim.add_var('ncells_capacity', Type_Int, 100)
        self.dim_ncells = self.sim.add_static_array('dim_cells', self.sim.ndims(), Type_Int)
        self.cell_capacity = self.sim.add_var('cell_capacity', Type_Int, 20)
        self.cell_particles = self.sim.add_array('cell_particles', [self.ncells_capacity, self.cell_capacity], Type_Int)
        self.cell_sizes = self.sim.add_array('cell_sizes', self.ncells_capacity, Type_Int)
        self.stencil = self.sim.add_array('stencil', self.nstencil_max, Type_Int)
        self.particle_cell = self.sim.add_array('particle_cell', self.sim.particle_capacity, Type_Int)


class CellListsStencilBuild:
    def __init__(self, cell_lists):
        self.cell_lists = cell_lists

    def lower(self):
        cl = self.cell_lists
        grid = cl.sim.grid
        index = None
        nall = 1

        cl.sim.clear_block()
        cl.sim.add_statement(Print(cl.sim, "CellListsStencilBuild"))

        for d in range(cl.sim.ndims()):
            cl.dim_ncells[d].set(Ceil(cl.sim, (grid.max(d) - grid.min(d)) / cl.spacing[d]) + 2)
            nall *= cl.dim_ncells[d]

        cl.ncells.set(nall)
        for resize in Resize(cl.sim, cl.ncells_capacity):
            for _ in Filter(cl.sim, cl.ncells >= cl.ncells_capacity):
                resize.set(cl.ncells)

        for _ in cl.sim.nest_mode():
            cl.nstencil.set(0)
            for d in range(cl.sim.ndims()):
                nneigh = cl.nneighbor_cells[d]
                for d_idx in For(cl.sim, -nneigh, nneigh + 1):
                    index = (d_idx if index is None else index * cl.dim_ncells[d - 1] + d_idx)
                    if d == cl.sim.ndims() - 1:
                        cl.stencil[cl.nstencil].set(index)
                        cl.nstencil.set(cl.nstencil + 1)

        return cl.sim.block


class CellListsBuild:
    def __init__(self, cell_lists):
        self.cell_lists = cell_lists

    def lower(self):
        cl = self.cell_lists
        grid = cl.sim.grid
        positions = cl.sim.property('position')

        cl.sim.clear_block()
        cl.sim.add_statement(Print(cl.sim, "CellListsBuild"))
        for resize in Resize(cl.sim, cl.cell_capacity):
            for c in For(cl.sim, 0, cl.ncells):
                cl.cell_sizes[c].set(0)

            for i in ParticleFor(cl.sim, local_only=False):
                cell_index = [
                    Cast.int(cl.sim, (positions[i][d] - grid.min(d)) / cl.spacing[d])
                    for d in range(0, cl.sim.ndims())]

                flat_idx = None
                for d in range(0, cl.sim.ndims()):
                    flat_idx = (cell_index[d] if flat_idx is None
                                else flat_idx * cl.dim_ncells[d] + cell_index[d])

                cell_size = cl.cell_sizes[flat_idx]
                for _ in Filter(cl.sim, BinOp.and_op(flat_idx >= 0, flat_idx <= cl.ncells)):
                    for cond in Branch(cl.sim, cell_size >= cl.cell_capacity):
                        if cond:
                            resize.set(cell_size)
                        else:
                            cl.cell_particles[flat_idx][cell_size].set(i)
                            cl.particle_cell[i].set(flat_idx)

                    cl.cell_sizes[flat_idx].set(cell_size + 1)

        return cl.sim.block