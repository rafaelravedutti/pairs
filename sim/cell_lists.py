from ast.block import BlockAST
from ast.branches import BranchAST
from ast.cast import CastAST
from ast.data_types import Type_Int
from ast.expr import ExprAST
from ast.loops import ForAST, ParticleForAST
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


class CellListsStencilBuild:
    def __init__(self, sim, cell_lists):
        self.sim = sim
        self.cell_lists = cell_lists

    def lower(self):
        dims = self.sim.dimensions
        cl = self.cell_lists
        loops = []
        index = None

        for d in range(0, dims):
            nneigh = cl.nneighbor_cells[d]
            loops.append(ForAST(self.sim, -nneigh, nneigh + 1))

            if d > 0:
                loops[d - 1].set_body(BlockAST(self.sim, [loops[d]]))

            index = (loops[d].iter() if index is None
                     else index * cl.ncells[d - 1] + loops[d].iter())

        loops[dims - 1].set_body(
            BlockAST(self.sim, [
                cl.stencil[cl.nstencil].set(index),
                cl.nstencil.set(cl.nstencil + 1)]))

        return BlockAST(self.sim, [cl.nstencil.set(0), loops[0]])


class CellListsBuild:
    def __init__(self, sim, cell_lists):
        self.sim = sim
        self.cell_lists = cell_lists

    def lower(self):
        cl = self.cell_lists
        cfg = cl.sim.grid_config
        positions = self.sim.property('position')
        reset_loop = ForAST(self.sim, 0, cl.ncells_all)
        reset_loop.set_body(BlockAST(self.sim,
                            [cl.cell_sizes[reset_loop.iter()].set(0)]))

        fill_loop = ParticleForAST(self.sim)
        cell_index = [
            CastAST.int(self.sim,
                        (positions[fill_loop.iter()][d] - cfg[d][0]) /
                        cl.spacing)
            for d in range(0, self.sim.dimensions)]

        flat_index = None
        for d in range(0, self.sim.dimensions):
            flat_index = (cell_index[d] if flat_index is None
                          else flat_index * cl.ncells[d] + cell_index[d])

        cell_size = cl.cell_sizes[flat_index]
        resize = Resize(self.sim, cl.cell_capacity, cl.cell_particles,
                        [reset_loop, fill_loop])

        fill_loop.set_body(BlockAST(self.sim, [
            BranchAST.if_stmt(self.sim, ExprAST.and_op(
                flat_index >= 0, flat_index <= cl.ncells_all), [
                    resize.check(cell_size, [
                        cl.cell_particles[flat_index][cell_size].set(
                            fill_loop.iter())]),
                    cl.cell_sizes[flat_index].set(cell_size + 1)])]))

        return resize.block()
