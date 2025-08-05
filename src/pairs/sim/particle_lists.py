from pairs.ir.assign import Assign
from pairs.ir.ast_term import ASTTerm
from pairs.ir.atomic import AtomicAdd
from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
from pairs.ir.loops import For, ParticleFor, While, Break
from pairs.ir.math import Ceil
from pairs.ir.scalars import ScalarOp
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.ir.print import Print
from pairs.sim.flags import Flags
from pairs.sim.lowerable import Lowerable

class ParticleLists:
    def __init__(self, sim):
        self.sim = sim
        self.shape_nparticles   =   self.sim.add_array('shape_nparticles', self.sim.max_shapes(), Types.Int32)
        self.shape_partitioned_idx = self.sim.add_array('shape_partitioned_idx', self.sim.particle_capacity, Types.Int32)


class BuildShapePartitions(Lowerable):
    def __init__(self, sim, particle_lists, cell_lists):
        super().__init__(sim)
        self.particle_lists = particle_lists
        self.cell_lists = cell_lists

    # TODO: Implement this module for device
    @pairs_host_block
    def lower(self):
        self.sim.module_name("build_shape_partitions")
        shape_nparticles = self.particle_lists.shape_nparticles
        shape_partitioned_idx = self.particle_lists.shape_partitioned_idx
        shapes_buffer = self.cell_lists.shapes_buffer

        idx = self.sim.add_temp_var(0)
        for shape in For(self.sim, 0, self.sim.max_shapes()):
            for i in For(self.sim, 0, self.sim.nlocal):
                for _ in Filter(self.sim, ScalarOp.cmp(self.sim.particle_shape[i], shapes_buffer[shape])):
                    Assign(self.sim, shape_partitioned_idx[idx], i)
                    Assign(self.sim, idx, idx + 1)

            Assign(self.sim, shape_nparticles[shape], idx)


