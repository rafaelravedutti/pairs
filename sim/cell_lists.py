from ast.bin_op import BinOp
from ast.branches import Branch, Filter
from ast.cast import Cast
from ast.data_types import Type_Int
from ast.loops import For, ParticleFor
from ast.utils import Print
from functools import reduce
from sim.resize import Resize
import math


class CellLists:
    def __init__(self, sim, grid, spacing, cutoff_radius):
        self.sim = sim
        self.grid = grid
        self.spacing = spacing

        self.nneighbor_cells = [
            math.ceil(cutoff_radius / (
                spacing if not isinstance(spacing, list)
                else spacing[d]))
            for d in range(0, sim.dimensions)]

        self.nstencil = self.sim.add_var('nstencil', Type_Int)
        self.nstencil_max = reduce((lambda x, y: x * y), [
            self.nneighbor_cells[d] * 2 + 1 for d in range(0, sim.dimensions)])

        self.ncells_all = self.sim.add_var('ncells_all', Type_Int)
        self.cell_capacity = self.sim.add_var('cell_capacity', Type_Int, 20)
        self.ncells = self.sim.add_static_array('ncells', self.sim.dimensions, Type_Int)
        self.cell_particles = self.sim.add_array('cell_particles', [self.ncells_all, self.cell_capacity], Type_Int)
        self.cell_sizes = self.sim.add_array('cell_sizes', self.ncells_all, Type_Int)
        self.stencil = self.sim.add_array('stencil', self.nstencil_max, Type_Int)
        self.particle_cell = self.sim.add_array('particle_cell', self.sim.particle_capacity, Type_Int)


class CellListsStencilBuild:
    def __init__(self, cell_lists):
        self.cell_lists = cell_lists

    def lower(self):
        cl = self.cell_lists
        ncells = cl.sim.grid.get_ncells_for_spacing(cl.spacing)
        index = None

        cl.sim.clear_block()
        cl.sim.add_statement(Print(cl.sim, "CellListsStencilBuild"))

        nall = 1
        for d in range(0, cl.sim.dimensions):
            cl.ncells[d].set(ncells[d])
            nall *= ncells[d]

        cl.ncells_all.set_initial_value(nall)

        for _ in cl.sim.nest_mode():
            cl.nstencil.set(0)
            for d in range(0, cl.sim.dimensions):
                nneigh = cl.nneighbor_cells[d]
                for d_idx in For(cl.sim, -nneigh, nneigh + 1):
                    index = (d_idx if index is None
                             else index * cl.ncells[d - 1] + d_idx)

                    if d == cl.sim.dimensions - 1:
                        cl.stencil[cl.nstencil].set(index)
                        cl.nstencil.set(cl.nstencil + 1)

        return cl.sim.block


class CellListsBuild:
    def __init__(self, cell_lists):
        self.cell_lists = cell_lists

    def lower(self):
        cl = self.cell_lists
        grid = cl.sim.grid
        spc = cl.spacing
        positions = cl.sim.property('position')

        cl.sim.clear_block()
        cl.sim.add_statement(Print(cl.sim, "CellListsBuild"))
        for resize in Resize(cl.sim, cl.cell_capacity):
            for c in For(cl.sim, 0, cl.ncells_all):
                cl.cell_sizes[c].set(0)

            for i in ParticleFor(cl.sim, local_only=False):
                cell_index = [
                    Cast.int(cl.sim, (positions[i][d] - grid.min(d)) / spc)
                    for d in range(0, cl.sim.dimensions)]

                flat_idx = None
                for d in range(0, cl.sim.dimensions):
                    flat_idx = (cell_index[d] if flat_idx is None
                                else flat_idx * cl.ncells[d] + cell_index[d])

                cell_size = cl.cell_sizes[flat_idx]
                for _ in Filter(cl.sim, BinOp.and_op(flat_idx >= 0, flat_idx <= cl.ncells_all)):
                    for cond in Branch(cl.sim, cell_size >= cl.cell_capacity):
                        if cond:
                            resize.set(cell_size)
                        else:
                            cl.cell_particles[flat_idx][cell_size].set(i)
                            cl.particle_cell[i].set(flat_idx)

                    cl.cell_sizes[flat_idx].set(cell_size + 1)

        return cl.sim.block
