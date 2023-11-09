from functools import reduce
import math
from pairs.ir.assign import Assign
from pairs.ir.ast_term import ASTTerm
from pairs.ir.atomic import AtomicAdd
from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
from pairs.ir.loops import For, ParticleFor, While
from pairs.ir.math import Ceil
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.flags import Flags
from pairs.sim.lowerable import Lowerable


class CellLists:
    def __init__(self, sim, grid, spacing, cutoff_radius):
        self.sim = sim
        self.grid = grid
        self.spacing = spacing if isinstance(spacing, list) else [spacing for d in range(sim.ndims())]
        self.cutoff_radius = cutoff_radius
        self.nneighbor_cells = [math.ceil(cutoff_radius / self.spacing[d]) for d in range(sim.ndims())]
        self.nstencil_max = reduce((lambda x, y: x * y), [self.nneighbor_cells[d] * 2 + 1 for d in range(sim.ndims())])
        # Data introduced in the simulation
        self.nstencil           =   self.sim.add_var('nstencil', Types.Int32)
        self.ncells             =   self.sim.add_var('ncells', Types.Int32, 1)
        self.ncells_capacity    =   self.sim.add_var('ncells_capacity', Types.Int32, 100)
        self.cell_capacity      =   self.sim.add_var('cell_capacity', Types.Int32, 20)
        self.dim_ncells         =   self.sim.add_static_array('dim_cells', self.sim.ndims(), Types.Int32)
        self.shapes_buffer      =   self.sim.add_static_array('shapes_buffer', self.sim.max_shapes(), Types.Int32)
        self.cell_particles     =   self.sim.add_array('cell_particles', [self.ncells_capacity, self.cell_capacity], Types.Int32)
        self.cell_sizes         =   self.sim.add_array('cell_sizes', self.ncells_capacity, Types.Int32)
        self.nshapes            =   self.sim.add_array('nshapes', [self.ncells_capacity, self.sim.max_shapes()], Types.Int32)
        self.stencil            =   self.sim.add_array('stencil', self.nstencil_max, Types.Int32)
        self.particle_cell      =   self.sim.add_array('particle_cell', self.sim.particle_capacity, Types.Int32)


class BuildCellListsStencil(Lowerable):
    def __init__(self, sim, cell_lists):
        super().__init__(sim)
        self.cell_lists = cell_lists

    @pairs_host_block
    def lower(self):
        sim = self.sim
        cl = self.cell_lists
        index = None
        ntotal_cells = 1

        sim.module_name("build_cell_lists_stencil")
        sim.check_resize(cl.ncells_capacity, cl.ncells)

        for d in range(sim.ndims()):
            dmin = sim.grid.min(d) - cl.spacing[d]
            dmax = sim.grid.max(d) + cl.spacing[d]
            Assign(sim, cl.dim_ncells[d], Ceil(sim, (dmax - dmin) / cl.spacing[d]) + 1)
            ntotal_cells *= cl.dim_ncells[d]

        Assign(sim, cl.ncells, ntotal_cells + 1)

        for _ in sim.nest_mode():
            Assign(sim, cl.nstencil, 0)

            for d in range(sim.ndims()):
                nneigh = cl.nneighbor_cells[d]

                for d_idx in For(sim, -nneigh, nneigh + 1):
                    index = (d_idx if index is None else index + cl.dim_ncells[d - 1] * d_idx)
                    if d == sim.ndims() - 1:
                        Assign(sim, cl.stencil[cl.nstencil], index)
                        Assign(sim, cl.nstencil, cl.nstencil + 1)


class BuildCellLists(Lowerable):
    def __init__(self, sim, cell_lists):
        super().__init__(sim)
        self.cell_lists = cell_lists

    @pairs_device_block
    def lower(self):
        sim = self.sim
        cl = self.cell_lists
        particle_flags = sim.particle_flags
        positions = sim.position()
        sim.module_name("build_cell_lists")
        sim.check_resize(cl.cell_capacity, cl.cell_sizes)

        for c in For(sim, 0, cl.ncells):
            Assign(sim, cl.cell_sizes[c], 0)

        for i in ParticleFor(sim, local_only=False):
            flat_index = sim.add_temp_var(0)

            for _ in Filter(sim, ASTTerm.not_op(particle_flags[i] & Flags.Infinite)):
                cell_index = [
                    Cast.int(sim, (positions[i][d] - (sim.grid.min(d) - cl.spacing[d])) / cl.spacing[d]) \
                    for d in range(sim.ndims())]
                index_1d = None

                for d in range(sim.ndims()):
                    index_1d = (cell_index[d] if index_1d is None else index_1d + cl.dim_ncells[d - 1] * cell_index[d])

                Assign(sim, flat_index, index_1d + 1)

            for _ in Filter(sim, ScalarOp.and_op(flat_index >= 0, flat_index < cl.ncells)):
                index_in_cell = AtomicAdd(sim, cl.cell_sizes[flat_index], 1)
                Assign(sim, cl.particle_cell[i], flat_index)
                Assign(sim, cl.cell_particles[flat_index][index_in_cell], i)


class PartitionCellLists(Lowerable):
    def __init__(self, sim, cell_lists):
        super().__init__(sim)
        self.cell_lists = cell_lists

    @pairs_device_block
    def lower(self):
        self.sim.module_name("partition_cell_lists")
        cell_particles = self.cell_lists.cell_particles
        shapes_buffer = self.cell_lists.shapes_buffer

        for s in range(self.sim.max_shapes()):
            Assign(self.sim, shapes_buffer[s], self.sim.get_shape_id(s))

        for cell in For(self.sim, 0, self.cell_lists.ncells):
            start = self.sim.add_temp_var(0)
            end = self.sim.add_temp_var(0)

            for shape in For(self.sim, 0, self.sim.max_shapes()):
                shape_id = shapes_buffer[shape]
                shape_start = self.sim.add_temp_var(start)
                Assign(self.sim, end, self.cell_lists.cell_sizes[cell] - 1)

                for _ in While(self.sim, start <= end):
                    particle = cell_particles[cell][start]

                    for unmatch in Branch(self.sim, ScalarOp.neq(self.sim.particle_shape[particle], shape_id)):
                        if unmatch:
                            for _ in Filter(self.sim, ScalarOp.neq(start, end)):
                                Assign(self.sim, cell_particles[cell][start], cell_particles[cell][end])
                                Assign(self.sim, cell_particles[cell][end], particle)

                            Assign(self.sim, end, end - 1)

                        else:
                            Assign(self.sim, start, start + 1)
                            Assign(self.sim, self.cell_lists.nshapes[cell][shape], start - shape_start)
