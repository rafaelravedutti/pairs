from functools import reduce
import math
from pairs.ir.assign import Assign
from pairs.ir.ast_term import ASTTerm
from pairs.ir.atomic import AtomicAdd
from pairs.ir.scalars import ScalarOp
from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
from pairs.ir.loops import For, ParticleFor
from pairs.ir.math import Ceil
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
        self.cell_particles     =   self.sim.add_array('cell_particles', [self.ncells_capacity, self.cell_capacity], Types.Int32)
        self.cell_sizes         =   self.sim.add_array('cell_sizes', self.ncells_capacity, Types.Int32)
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
        grid = sim.grid
        index = None
        ntotal_cells = 1

        sim.module_name("build_cell_lists_stencil")
        sim.check_resize(cl.ncells_capacity, cl.ncells)

        for d in range(sim.ndims()):
            Assign(sim, cl.dim_ncells[d], Ceil(sim, (grid.max(d) - grid.min(d)) / cl.spacing[d]) + 2)
            ntotal_cells *= cl.dim_ncells[d]

        Assign(sim, cl.ncells, ntotal_cells + 1)

        for _ in sim.nest_mode():
            Assign(sim, cl.nstencil, 0)

            for d in range(sim.ndims()):
                nneigh = cl.nneighbor_cells[d]

                for d_idx in For(sim, -nneigh, nneigh + 1):
                    index = (d_idx if index is None else index * cl.dim_ncells[d - 1] + d_idx)
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
        grid = sim.grid
        particle_flags = sim.particle_flags
        positions = sim.position()
        sim.module_name("build_cell_lists")
        sim.check_resize(cl.cell_capacity, cl.cell_sizes)

        for c in For(sim, 0, cl.ncells):
            Assign(sim, cl.cell_sizes[c], 0)

        for i in ParticleFor(sim, local_only=False):
            flat_index = sim.add_temp_var(0)

            for _ in Filter(sim, ASTTerm.not_op(particle_flags[i] & Flags.Infinite)):
                cell_index = [Cast.int(sim, (positions[i][d] - grid.min(d)) / cl.spacing[d]) for d in range(sim.ndims())]
                index_1d = None

                for d in range(sim.ndims()):
                    index_1d = (cell_index[d] if index_1d is None else index_1d * cl.dim_ncells[d] + cell_index[d])

                Assign(sim, flat_index, index_1d + 1)

            for _ in Filter(sim, ScalarOp.and_op(flat_index >= 0, flat_index < cl.ncells)):
                index_in_cell = AtomicAdd(sim, cl.cell_sizes[flat_index], 1)
                Assign(sim, cl.particle_cell[i], flat_index)
                Assign(sim, cl.cell_particles[flat_index][index_in_cell], i)
