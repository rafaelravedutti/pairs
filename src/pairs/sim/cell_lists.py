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
    def __init__(self, sim, dom_part, spacing, cutoff_radius):
        self.sim = sim
        self.dom_part = dom_part
        self.spacing = spacing if isinstance(spacing, list) else [spacing for d in range(sim.ndims())]
        self.cutoff_radius = cutoff_radius
        self.nneighbor_cells = [math.ceil(cutoff_radius / self.spacing[d]) for d in range(sim.ndims())]
        self.nstencil_max = reduce((lambda x, y: x * y), [self.nneighbor_cells[d] * 2 + 1 for d in range(sim.ndims())])
        # Data introduced in the simulation
        self.nstencil           =   self.sim.add_var('nstencil', Types.Int32)
        self.ncells             =   self.sim.add_var('ncells', Types.Int32, 1)
        self.ncells_capacity    =   self.sim.add_var('ncells_capacity', Types.Int32, 100000)
        self.cell_capacity      =   self.sim.add_var('cell_capacity', Types.Int32, 64)
        self.dim_ncells         =   self.sim.add_array('dim_cells', self.sim.ndims(), Types.Int32)
        self.shapes_buffer      =   self.sim.add_array('shapes_buffer', self.sim.max_shapes(), Types.Int32)
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
        stencil = self.cell_lists.stencil
        nstencil = self.cell_lists.nstencil
        spacing = self.cell_lists.spacing
        nneighbor_cells = self.cell_lists.nneighbor_cells
        dim_ncells = self.cell_lists.dim_ncells
        ncells = self.cell_lists.ncells
        ncells_capacity = self.cell_lists.ncells_capacity
        shapes_buffer = self.cell_lists.shapes_buffer
        index = None
        ntotal_cells = 1

        self.sim.module_name("build_cell_lists_stencil")
        self.sim.check_resize(ncells_capacity, ncells)

        for s in range(self.sim.max_shapes()):
            Assign(self.sim, shapes_buffer[s], self.sim.get_shape_id(s))

        for dim in range(self.sim.ndims()):
            dim_min = self.cell_lists.dom_part.min(dim) - spacing[dim]
            dim_max = self.cell_lists.dom_part.max(dim) + spacing[dim]
            Assign(self.sim, dim_ncells[dim], Ceil(self.sim, (dim_max - dim_min) / spacing[dim]) + 1)
            ntotal_cells *= dim_ncells[dim]

        Assign(self.sim, ncells, ntotal_cells + 1)

        for _ in self.sim.nest_mode():
            Assign(self.sim, nstencil, 0)

            for dim in range(self.sim.ndims()):
                nneigh = nneighbor_cells[dim]
                for dim_offset in For(self.sim, -nneigh, nneigh + 1):
                    index = dim_offset if index is None else index * dim_ncells[dim] + dim_offset
                    if dim == self.sim.ndims() - 1:
                        Assign(self.sim, stencil[nstencil], index)
                        Assign(self.sim, nstencil, nstencil + 1)


class BuildCellLists(Lowerable):
    def __init__(self, sim, cell_lists):
        super().__init__(sim)
        self.cell_lists = cell_lists

    @pairs_device_block
    def lower(self):
        particle_flags = self.sim.particle_flags
        particle_cell = self.cell_lists.particle_cell
        cell_particles = self.cell_lists.cell_particles
        cell_sizes = self.cell_lists.cell_sizes
        cell_capacity = self.cell_lists.cell_capacity
        spacing = self.cell_lists.spacing
        dim_ncells = self.cell_lists.dim_ncells
        ncells = self.cell_lists.ncells
        dom_part = self.cell_lists.dom_part
        positions = self.sim.position()

        self.sim.module_name("build_cell_lists")
        self.sim.check_resize(cell_capacity, cell_sizes)

        for c in For(self.sim, 0, ncells):
            Assign(self.sim, cell_sizes[c], 0)

        for i in ParticleFor(self.sim, local_only=False):
            flat_index = self.sim.add_temp_var(0)

            for _ in Filter(self.sim, ASTTerm.not_op(particle_flags[i] & Flags.Infinite)):
                cell_index = [
                    Cast.int(self.sim,
                        (positions[i][dim] - (dom_part.min(dim) - spacing[dim])) / spacing[dim]) \
                    for dim in range(self.sim.ndims())]

                index = None
                for dim in range(self.sim.ndims()):
                    index = cell_index[dim] if index is None \
                            else index * dim_ncells[dim] + cell_index[dim]

                Assign(self.sim, flat_index, index + 1)

            for _ in Filter(self.sim, ScalarOp.and_op(flat_index >= 0, flat_index < ncells)):
                index_in_cell = AtomicAdd(self.sim, cell_sizes[flat_index], 1)
                Assign(self.sim, particle_cell[i], flat_index)
                Assign(self.sim, cell_particles[flat_index][index_in_cell], i)


class PartitionCellLists(Lowerable):
    def __init__(self, sim, cell_lists):
        super().__init__(sim)
        self.cell_lists = cell_lists

    @pairs_device_block
    def lower(self):
        self.sim.module_name("partition_cell_lists")
        cell_particles = self.cell_lists.cell_particles
        shapes_buffer = self.cell_lists.shapes_buffer

        for cell in For(self.sim, 0, self.cell_lists.ncells):
            start = self.sim.add_temp_var(0)
            end = self.sim.add_temp_var(0)

            for shape in For(self.sim, 0, self.sim.max_shapes()):
                shape_id = shapes_buffer[shape]
                shape_start = self.sim.add_temp_var(start)
                Assign(self.sim, end, self.cell_lists.cell_sizes[cell] - 1)
                Assign(self.sim, self.cell_lists.nshapes[cell][shape], 0)

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
