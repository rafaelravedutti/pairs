from ast.branches import Branch, Filter
from ast.cast import Cast
from ast.data_types import Type_Int
from ast.expr import Expr
from ast.loops import For, ParticleFor
from functools import reduce
from sim.resize import Resize
import math


class CellLists:
    def __init__(self, sim, spacing, cutoff_radius):
        self.sim = sim
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
        self.cell_capacity = self.sim.add_var('cell_capacity', Type_Int)
        self.ncells = self.sim.add_array(
            'ncells', self.sim.dimensions, Type_Int)
        self.cell_particles = self.sim.add_array(
            'cell_particles', [self.ncells_all, self.cell_capacity], Type_Int)
        self.cell_sizes = self.sim.add_array(
            'cell_sizes', self.ncells_all, Type_Int)
        self.stencil = self.sim.add_array(
            'stencil', self.nstencil_max, Type_Int)
        self.particle_cell = self.sim.add_array(
            'particle_cell', self.sim.nparticles, Type_Int)


class CellListsStencilBuild:
    def __init__(self, sim, cell_lists):
        self.sim = sim
        self.cell_lists = cell_lists

    def lower(self):
        cl = self.cell_lists
        index = None

        self.sim.clear_block()
        for _ in self.sim.nest_mode():
            cl.nstencil.set(0)
            for d in range(0, self.sim.dimensions):
                nneigh = cl.nneighbor_cells[d]
                for d_idx in For(self.sim, -nneigh, nneigh + 1):
                    index = (d_idx if index is None
                             else index * cl.ncells[d - 1] + d_idx)

                    if d == self.sim.dimensions - 1:
                        cl.stencil[cl.nstencil].set(index)
                        cl.nstencil.set(cl.nstencil + 1)

        return self.sim.block


class CellListsBuild:
    def __init__(self, sim, cell_lists):
        self.sim = sim
        self.cell_lists = cell_lists

    def lower(self):
        cl = self.cell_lists
        cfg = cl.sim.grid_config
        spc = cl.spacing
        positions = self.sim.property('position')

        self.sim.clear_block()
        for cap, rsz in Resize(self.sim, cl.cell_capacity, cl.cell_particles):
            for c in For(self.sim, 0, cl.ncells_all):
                cl.cell_sizes[c].set(0)

            for i in ParticleFor(self.sim):
                cell_index = [
                    Cast.int(self.sim, (positions[i][d] - cfg[d][0]) / spc)
                    for d in range(0, self.sim.dimensions)]

                flat_idx = None
                for d in range(0, self.sim.dimensions):
                    flat_idx = (cell_index[d] if flat_idx is None
                                else flat_idx * cl.ncells[d] + cell_index[d])

                cell_size = cl.cell_sizes[flat_idx]
                for _ in Filter(self.sim,
                                   Expr.and_op(flat_idx >= 0,
                                                  flat_idx <= cl.ncells_all)):
                    for cond in Branch(self.sim, cell_size > cap):
                        if cond:
                            rsz.set(cell_size)
                        else:
                            cl.cell_particles[flat_idx][cell_size].set(i)
                            cl.particle_cell[i].set(flat_idx)

                    cl.cell_sizes[flat_idx].set(cell_size + 1)

        return self.sim.block
