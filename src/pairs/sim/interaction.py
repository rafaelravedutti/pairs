from pairs.ir.assign import Assign
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.block import Block, pairs_inline
from pairs.ir.branches import Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.math import Sqrt
from pairs.ir.types import Types
from pairs.ir.vectors import Vector
from pairs.sim.flags import Flags
from pairs.sim.lowerable import Lowerable
from pairs.sim.shapes import Shapes


class Neighbor(ASTTerm):
    def __init__(self, sim, neighbor_index, cell_id, particle_index, shape):
        super().__init__(sim, ScalarOp)
        self._neighbor_index = neighbor_index
        self._cell_id = cell_id
        self._particle_index = particle_index
        self._shape = shape

    def __str__(self):
        return f"Neighbor<{self._neighbor_index}, {self._cell_id}>"

    def type(self):
        return Types.Int32

    def neighbor_index(self):
        return self._neighbor_index

    def cell_id(self):
        return self._cell_id

    def particle_index(self):
        return self._particle_index

    def shape(self):
        return self._shape


class NeighborFor:
    def number_of_cases(sim, use_cell_lists):
        return 2 if sim.neighbor_lists is None or use_cell_lists else 1

    def __init__(self, sim, particle, cell_lists, neighbor_lists=None):
        self.sim = sim
        self.particle = particle
        self.cell_lists = cell_lists
        self.neighbor_lists = neighbor_lists

    def __str__(self):
        return f"NeighborFor<{self.particle}>"

    def __iter__(self):
        if self.neighbor_lists is None:
            stencil = self.cell_lists.stencil
            nstencil = self.cell_lists.nstencil
            ncells = self.cell_lists.ncells
            particle_cell = self.cell_lists.particle_cell
            cell_particles = self.cell_lists.cell_particles
            particle_shape = self.sim.particle_shape
            nshapes = self.cell_lists.nshapes

            for shape in range(self.sim.max_shapes()):
                for disp in For(self.sim, 0, self.cell_lists.nstencil):
                    neigh_cell = particle_cell[self.particle] + stencil[disp]

                    for _ in Filter(self.sim, ScalarOp.and_op(neigh_cell > 0, neigh_cell < ncells)):
                        start = sum([nshapes[neigh_cell][s] for s in range(shape)], 0)
                        for cell_particle in For(self.sim, start, start + nshapes[neigh_cell][shape]):
                            particle_id = cell_particles[neigh_cell][cell_particle]

                            if self.sim._compute_half:
                                shape_cond = particle_shape[particle_id] > shape
                                condition = ScalarOp.or_op(shape_cond,
                                                           ScalarOp.and_op(ScalarOp.not_op(shape_cond),
                                                                           self.particle < particle_id))

                            else:
                                condition = ScalarOp.neq(particle_id, self.particle)

                            for _ in Filter(self.sim, condition):
                                yield Neighbor(self.sim, cell_particle, neigh_cell, particle_id, shape)

                # Infinite particles
                start = sum([nshapes[0][s] for s in range(shape)], 0)
                for inf_id in For(self.sim, start, start + nshapes[0][shape]):
                    inf_particle = cell_particles[0][inf_id]

                    if self.sim._compute_half:
                        shape_cond = particle_shape[inf_particle] > shape
                        condition = ScalarOp.or_op(shape_cond,
                                                   ScalarOp.and_op(ScalarOp.not_op(shape_cond),
                                                                   self.particle < inf_particle))

                    else:
                        condition = ScalarOp.neq(inf_particle, self.particle)

                    for _ in Filter(self.sim, condition):
                        yield Neighbor(self.sim, inf_id, 0, inf_particle, shape)

        else:
            neighborlists = self.neighbor_lists.neighborlists
            numneighs = self.neighbor_lists.numneighs

            for shape in range(self.sim.max_shapes()):
                start = sum([numneighs[self.particle][s] for s in range(shape)], 0)
                for k in For(self.sim, start, start + numneighs[self.particle][shape]):
                    yield Neighbor(self.sim, k, None, neighborlists[self.particle][k], shape)


class InteractionData:
    def __init__(self, sim, shape):
        self._i = sim.add_symbol(Types.Int32)
        self._j = sim.add_symbol(Types.Int32)
        self._delta = sim.add_symbol(Types.Vector)
        self._squared_distance = sim.add_symbol(Types.Real)
        self._penetration_depth = sim.add_symbol(Types.Real)
        self._contact_point = sim.add_symbol(Types.Vector)
        self._contact_normal = sim.add_symbol(Types.Vector)
        self._shape = shape

    def i(self):
        return self._i

    def j(self):
        return self._j

    def delta(self):
        return self._delta

    def squared_distance(self):
        return self._squared_distance

    def penetration_depth(self):
        return self._penetration_depth

    def contact_point(self):
        return self._contact_point

    def contact_normal(self):
        return self._contact_normal

    def shape(self):
        return self._shape


class ParticleInteraction(Lowerable):
    def __init__(self, sim, nbody, cutoff_radius, use_cell_lists=False):
        super().__init__(sim)
        self.nbody = nbody
        self.cutoff_radius = cutoff_radius
        self.contact_threshold = 0.0
        self.use_cell_lists = use_cell_lists
        self.ncases = NeighborFor.number_of_cases(sim, use_cell_lists)
        self.interactions_data = []
        self.blocks = [Block(sim, []) for _ in range(sim.max_shapes() * self.ncases)]
        self.active_block = None
        self.apply_list = set()

    def add_statement(self, stmt):
        self.active_block.add_statement(stmt)

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)
        self.sim.use_apply_list(self.apply_list)

        # Neighbors vary across iterations
        for shape in range(self.sim.max_shapes()):
            for case in range(self.ncases):
                self.active_block = self.blocks[shape * self.ncases + case]
                self.interactions_data.append(InteractionData(self.sim, shape))
                yield self.interactions_data[-1]

        self.sim.release_apply_list()
        self.sim.leave()
        self.active_block = None

    @pairs_inline
    def lower(self):
        if self.nbody == 2:
            position = self.sim.position()
            cell_lists = self.sim.cell_lists
            neighbor_lists = None if self.use_cell_lists else self.sim.neighbor_lists

            for i in ParticleFor(self.sim):
                for _ in Filter(self.sim, ScalarOp.cmp(self.sim.particle_flags[i] & Flags.Fixed, 0)):
                    for app in self.apply_list:
                        app.add_reduction_variable()

                    interaction = 0
                    for neigh in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                        interaction_data = self.interactions_data[interaction]
                        shape = interaction_data.shape()
                        shape_id = self.sim.get_shape_id(shape)
                        j = neigh.particle_index()

                        if shape_id == Shapes.PointMass:
                            delta = position[i] - position[j]
                            squared_distance = delta.x() * delta.x() + \
                                               delta.y() * delta.y() + \
                                               delta.z() * delta.z()
                            separation_dist = self.cutoff_radius
                            cutoff_condition = squared_distance < self.cutoff_radius
                            distance = Sqrt(self.sim, squared_distance)
                            penetration_depth = None
                            contact_normal = None
                            contact_point = None

                        elif shape_id == Shapes.Sphere:
                            radius = self.sim.property('radius')
                            delta = position[i] - position[j]
                            squared_distance = delta.x() * delta.x() + \
                                               delta.y() * delta.y() + \
                                               delta.z() * delta.z()
                            separation_dist = radius[i] + radius[j] + self.contact_threshold
                            cutoff_condition = squared_distance < separation_dist * separation_dist
                            distance = Sqrt(self.sim, squared_distance)
                            penetration_depth = distance - radius[i] - radius[j]
                            contact_normal = delta * (1.0 / distance)
                            k = radius[j] + 0.5 * penetration_depth
                            contact_point = position[j] + contact_normal * k

                        elif shape_id == Shapes.Halfspace:
                            radius = self.sim.property('radius')
                            normal = self.sim.property('normal')

                            d = normal[j][0] * position[j][0] + \
                                normal[j][1] * position[j][1] + \
                                normal[j][2] * position[j][2]

                            k = normal[j][0] * position[i][0] + \
                                normal[j][1] * position[i][1] + \
                                normal[j][2] * position[i][2]

                            penetration_depth = k - radius[i] - d
                            cutoff_condition = penetration_depth < self.contact_threshold
                            tmp = radius[i] + penetration_depth
                            contact_normal = normal[j]
                            contact_point = position[i] - Vector(self.sim, [tmp, tmp, tmp]) * normal[j]

                        else:
                            raise Exception("Invalid shape id.")

                        interaction_data.i().assign(i)
                        interaction_data.j().assign(j)
                        interaction_data.delta().assign(delta)
                        interaction_data.squared_distance().assign(squared_distance)
                        interaction_data.penetration_depth().assign(penetration_depth)
                        interaction_data.contact_point().assign(contact_point)
                        interaction_data.contact_normal().assign(contact_normal)
                        self.sim.add_statement(Filter(self.sim, cutoff_condition, self.blocks[interaction]))
                        interaction += 1

                    prop_reductions = {}
                    for app in self.apply_list:
                        prop = app.prop()
                        reduction = app.reduction_variable()

                        if prop not in prop_reductions:
                            prop_reductions[prop] = reduction

                        else:
                            prop_reductions[prop] = prop_reductions[prop] + reduction

                    for prop, reduction in prop_reductions.items():
                        Assign(self.sim, prop[i], reduction)

        else:
            raise Exception("Interactions among more than two particles are currently not supported.")
