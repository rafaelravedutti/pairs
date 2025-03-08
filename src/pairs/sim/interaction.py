from pairs.ir.assign import Assign
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.block import Block, pairs_device_block
from pairs.ir.branches import Filter, Branch
from pairs.ir.loops import For, ParticleFor
from pairs.ir.math import Sqrt, Abs, Min, Sign
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.ir.vectors import Vector
from pairs.ir.print import Print
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
    def __init__(self, sim, particle, cell_lists, neighbor_lists=None, shapes=None):
        self.sim = sim
        self.particle = particle
        self.cell_lists = cell_lists
        self.neighbor_lists = neighbor_lists
        # self.shapes = range(sim.max_shapes()) if shapes is None else shapes
        self.shapes = [shapes]

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

            if self.sim._store_neighbors_per_cell:
                cell_nneighs = self.cell_lists.cell_nneighs
                cell_neighbors = self.cell_lists.cell_neighbors

                for shape in self.shapes:
                    # FIXME: Without the inline, the 'cell' expression is being generated after
                    # its usage in the loop upper limit
                    cell = ScalarOp.inline(particle_cell[self.particle])
                    start = sum([cell_nneighs[cell][s] for s in range(shape)], 0)
                    for k in For(self.sim, start, start + cell_nneighs[cell][shape]):
                        particle_id = cell_neighbors[cell][k]

                        if self.sim._compute_half:
                            shape_id = self.sim.get_shape_id(shape)
                            condition = ScalarOp.or_op(
                                particle_shape[particle_id] > shape_id,
                                ScalarOp.and_op(
                                    ScalarOp.cmp(particle_shape[particle_id], shape_id),
                                    self.particle < particle_id))

                        else:
                            condition = ScalarOp.neq(particle_id, self.particle)

                        for _ in Filter(self.sim, condition):
                            yield Neighbor(self.sim, k, None, particle_id, shape)

            else:
                for shape in self.shapes:
                    for disp in For(self.sim, -1, self.cell_lists.nstencil):
                        neigh_cell = \
                            Select(self.sim, disp < 0, 0, particle_cell[self.particle] + stencil[disp])

                        for _ in Filter(self.sim, ScalarOp.or_op(disp < 0,
                                                                 ScalarOp.and_op(neigh_cell > 0,
                                                                                 neigh_cell < ncells))):

                            start = sum([nshapes[neigh_cell][s] for s in range(shape)], 0)
                            for cell_particle in For(self.sim,
                                                     start,
                                                     start + nshapes[neigh_cell][shape]):

                                particle_id = cell_particles[neigh_cell][cell_particle]

                                if self.sim._compute_half:
                                    shape_id = self.sim.get_shape_id(shape)
                                    condition = ScalarOp.or_op(
                                        particle_shape[particle_id] > shape_id,
                                        ScalarOp.and_op(
                                            ScalarOp.cmp(particle_shape[particle_id], shape_id),
                                            self.particle < particle_id))

                                else:
                                    condition = ScalarOp.neq(particle_id, self.particle)

                                for _ in Filter(self.sim, condition):
                                    yield Neighbor(
                                        self.sim, cell_particle, neigh_cell, particle_id, shape)

        else:
            neighborlists = self.neighbor_lists.neighborlists
            numneighs = self.neighbor_lists.numneighs

            for shape in self.shapes:
                start = sum([numneighs[self.particle][s] for s in range(shape)], 0)
                for k in For(self.sim, start, start + numneighs[self.particle][shape]):
                    yield Neighbor(self.sim, k, None, neighborlists[self.particle][k], shape)


class InteractionData:
    def __init__(self, sim, shape):
        self.sim = sim
        self._i = sim.add_symbol(Types.Int32)
        self._j = sim.add_symbol(Types.Int32)
        self._delta = sim.add_symbol(Types.Vector)
        self._squared_distance = sim.add_symbol(Types.Real)
        self._penetration_depth = sim.add_symbol(Types.Real)
        self._contact_point = sim.add_symbol(Types.Vector)
        self._contact_normal = sim.add_symbol(Types.Vector)
        self._shape = shape
        self.contact_threshold = 0.0
        self.cutoff_condition = None

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
    
    def pointmass_pointmass(self, i, j, cutoff_radius):
        position = self.sim.position()
        delta = position[i] - position[j]
        squared_distance = delta.x() * delta.x() + \
                        delta.y() * delta.y() + \
                        delta.z() * delta.z()
        separation_dist = cutoff_radius * cutoff_radius
        self.cutoff_condition = squared_distance < separation_dist

        self.delta().assign(delta)
        self.squared_distance().assign(squared_distance)

    def sphere_halfspace(self, i, j):
        position = self.sim.position()
        radius = self.sim.property('radius')
        normal = self.sim.property('normal')

        d = normal[j][0] * position[j][0] + \
            normal[j][1] * position[j][1] + \
            normal[j][2] * position[j][2]

        k = normal[j][0] * position[i][0] + \
            normal[j][1] * position[i][1] + \
            normal[j][2] * position[i][2]

        penetration_depth = k - radius[i] - d
        self.cutoff_condition = penetration_depth < self.contact_threshold
        tmp = radius[i] + penetration_depth
        contact_normal = normal[j]
        contact_point = position[i] - Vector(self.sim, [tmp, tmp, tmp]) * normal[j]
        
        self.penetration_depth().assign(penetration_depth)
        self.contact_point().assign(contact_point)
        self.contact_normal().assign(contact_normal)

    def sphere_sphere(self, i, j):
        position = self.sim.position()
        radius = self.sim.property('radius')
        delta = position[i] - position[j]
        squared_distance = delta.x() * delta.x() + \
                        delta.y() * delta.y() + \
                        delta.z() * delta.z()
        separation_dist = radius[i] + radius[j] + self.contact_threshold
        self.cutoff_condition = squared_distance < separation_dist * separation_dist
        distance = Sqrt(self.sim, squared_distance)
        penetration_depth = distance - radius[i] - radius[j]
        contact_normal = delta * (1.0 / distance)
        k = radius[j] + 0.5 * penetration_depth
        contact_point = position[j] + contact_normal * k

        self.delta().assign(delta)
        self.squared_distance().assign(squared_distance)
        self.penetration_depth().assign(penetration_depth)
        self.contact_point().assign(contact_point)
        self.contact_normal().assign(contact_normal)

    def sphere_box(self, i, j, s_relative=True):
        s = i   # Sphere
        b = j   # Box
        position = self.sim.position()
        edge_length = self.sim.property('edge_length')
        rotation_matrix = self.sim.property('rotation_matrix')
        radius = self.sim.property('radius')
        
        l = edge_length[b] * 0.5
        rm = rotation_matrix[b]
        delta = position[s] - position[b]
        
        # Distance to sphere in the box coodinate: p = rm.T * delta 
        p0 = delta[0]*rm[0] + delta[1]*rm[3] + delta[2]*rm[6]
        p1 = delta[0]*rm[1] + delta[1]*rm[4] + delta[2]*rm[7]
        p2 = delta[0]*rm[2] + delta[1]*rm[5] + delta[2]*rm[8]

        # Check if the sphere is outside the box. If yes, clamp p to box bounds
        outside0 =  ScalarOp.or_op( p0 <  -l[0], p0 >  l[0])
        p0 = ScalarOp.and_op(p0 >= -l[0], p0 <= l[0]) * p0 + (p0 < -l[0])*(-l[0]) + (p0 > l[0])*l[0]
        outside1 =  ScalarOp.or_op( p1 <  -l[1], p1 >  l[1])
        p1 = ScalarOp.and_op(p1 >= -l[1], p1 <= l[1]) * p1 + (p1 < -l[1])*(-l[1]) + (p1 > l[1])*l[1]
        outside2 =  ScalarOp.or_op( p2 <  -l[2], p2 >  l[2])
        p2 = ScalarOp.and_op(p2 >= -l[2], p2 <= l[2]) * p2 + (p2 < -l[2])*(-l[2]) + (p2 > l[2])*l[2]

        self.cutoff_condition = self.sim.add_temp_var(0)
        squared_distance = self.sim.add_temp_var(0.0)
        penetration_depth = self.sim.add_temp_var(0.0)
        contact_point = self.sim.add_temp_var([0.0, 0.0, 0.0])
        contact_normal = self.sim.add_temp_var([0.0, 0.0, 0.0])

        for outside in Branch(self.sim, ScalarOp.or_op(ScalarOp.or_op(outside0, outside1), outside2)):
            if outside:
                # Transfrom p to global coordinate: q = rm * p
                q = Vector(self.sim, [  rm[0]*p0 + rm[1]*p1 + rm[2]*p2, 
                                        rm[3]*p0 + rm[4]*p1 + rm[5]*p2,
                                        rm[6]*p0 + rm[7]*p1 + rm[8]*p2])
                
                # Normal away from the box
                n = delta - q
                Assign(self.sim, squared_distance, n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
                distance = Sqrt(self.sim, squared_distance)
                Assign(self.sim, penetration_depth, distance - radius[s])
                Assign(self.sim, contact_point, position[b] + q)
                Assign(self.sim, contact_normal, n * (1.0 / distance))
                Assign(self.sim, self.cutoff_condition, penetration_depth < self.contact_threshold)
            else:
                # Print(self.sim, "Sphere", s, "(", position[s][0], position[s][1], position[s][2], ") is inside box", b)
                # If sphere is inside the box find the closest face
                dist0 = l[0] - Abs(self.sim, p0)
                dist1 = l[1] - Abs(self.sim, p1)
                dist2 = l[2] - Abs(self.sim, p2)
                mindist = Min(self.sim, Min(self.sim, dist0, dist1), dist2)
                face = Select(self.sim, ScalarOp.cmp(mindist, dist0), 0, 
                              Select(self.sim,ScalarOp.cmp(mindist, dist1), 1, 2))

                n = self.sim.add_temp_var([0.0, 0.0, 0.0])
                # In this case, the normal away from the box is an axis-aligned unit vector in the box coordinate 
                # Assign(self.sim, n[face], Sign(self.sim, p[face]))   
                # FIXME: Issue with index generation (a Select node can't be an index for VectorAccess)
                #        hence the 3 lines below:
                Assign(self.sim, n[0], Select(self.sim, ScalarOp.cmp(face,0), Sign(self.sim, p0), 0.0))
                Assign(self.sim, n[1], Select(self.sim, ScalarOp.cmp(face,1), Sign(self.sim, p1), 0.0))
                Assign(self.sim, n[2], Select(self.sim, ScalarOp.cmp(face,2), Sign(self.sim, p2), 0.0))

                # Transofram n to global coordinates
                Assign(self.sim, contact_normal[0], rm[0]*n[0] + rm[1]*n[1] + rm[2]*n[2]) 
                Assign(self.sim, contact_normal[1], rm[3]*n[0] + rm[4]*n[1] + rm[5]*n[2]) 
                Assign(self.sim, contact_normal[2], rm[6]*n[0] + rm[7]*n[1] + rm[8]*n[2]) 

                Assign(self.sim, penetration_depth, - mindist - radius[s])
                Assign(self.sim, contact_point, position[s])
                Assign(self.sim, self.cutoff_condition, penetration_depth < self.contact_threshold)

        self.delta().assign(delta)
        self.squared_distance().assign(squared_distance)
        self.penetration_depth().assign(penetration_depth)
        self.contact_point().assign(contact_point)
        # Contact normal is always computed towards the sphere
        self.contact_normal().assign(contact_normal if s_relative else -contact_normal)

class ParticleInteraction(Lowerable):
    def __init__(self, sim, module_name,  nbody, cutoff_radius=None, use_cell_lists=False):
        super().__init__(sim)
        self.module_name = module_name
        self.nbody = nbody
        self.contact_threshold = 0.0
        self.use_cell_lists = use_cell_lists
        self.maxs = self.sim.max_shapes()
        self.interactions_data = {}
        self.cutoff_radius = cutoff_radius
        if any(self.sim.get_shape_id(s)==Shapes.PointMass for s in range(self.maxs)): 
            assert cutoff_radius is not None

        # We reserve n*n blocks and apply_lists for n shapes present in the system, but we only use the included i-j interactions
        self.blocks = [Block(sim, []) for _ in range(self.maxs * self.maxs)]
        self.apply_list = [set() for _ in range(self.maxs * self.maxs)]
        self.active_block = None
        self.cell_lists = self.sim.cell_lists

    def add_statement(self, stmt):
        self.active_block.add_statement(stmt)

    # Included interactions
    # -------------------------------------------------------------------
    #   i \ j   |   Sphere  |   Halfspace   |   PointMass   |  Box  |
    # -------------------------------------------------------------------
    # Sphere    |     1     |       1       |       0       |   1   |
    # Halfspace |     0     |       0       |       0       |   0   |
    # PointMass |     0     |       0       |       1       |   0   |
    # Box       |     1     |       0       |       0       |   0   |

    def include_interaction(self, ishape, jshape):
        id_i = self.sim.get_shape_id(ishape)
        id_j = self.sim.get_shape_id(jshape)
        return  (id_i == Shapes.PointMass and id_j == Shapes.PointMass) or \
                (id_i == Shapes.Sphere and id_j == Shapes.Sphere) or \
                (id_i == Shapes.Sphere and id_j == Shapes.Halfspace) or \
                (id_i == Shapes.Sphere and id_j == Shapes.Box) or \
                (id_i == Shapes.Box and id_j == Shapes.Sphere)                     
                    
    # Included kernels
    def include_shape(self, ishape):
        id_i = self.sim.get_shape_id(ishape)
        return  (id_i == Shapes.PointMass) or \
                (id_i == Shapes.Sphere) or \
                (id_i == Shapes.Box)

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)

        # Neighbors vary across iterations
        for ishape in range(self.sim.max_shapes()):
            for jshape in range(self.sim.max_shapes()):
                if self.include_interaction(ishape, jshape):
                    apply_list_id = ishape*self.maxs + jshape
                    self.sim.use_apply_list(self.apply_list[apply_list_id])
                    self.active_block = self.blocks[ishape*self.maxs + jshape]
                    self.interactions_data[ishape*self.maxs + jshape] = InteractionData(self.sim, ishape*self.maxs + jshape)
                    yield self.interactions_data[ishape*self.maxs + jshape]
                    self.sim.release_apply_list()

        self.sim.leave()
        self.active_block = None

    def apply_reductions(self, i, ishape, jshape):
        prop_reductions = {}
        for app in self.apply_list[ishape*self.maxs + jshape]:
            prop = app.prop()
            reduction = app.reduction_variable()
            if prop not in prop_reductions:
                prop_reductions[prop] = reduction
            else:
                prop_reductions[prop] = prop_reductions[prop] + reduction

        for prop, reduction in prop_reductions.items():
            Assign(self.sim, prop[i], prop[i] + reduction)

    def compute_interaction(self, i, j, ishape, jshape):
        interaction_data = self.interactions_data[ishape*self.maxs + jshape]
        interaction_data.i().assign(i)
        interaction_data.j().assign(j)
        ishape_id = self.sim.get_shape_id(ishape)
        jshape_id = self.sim.get_shape_id(jshape)
        
        if ishape_id == Shapes.PointMass and jshape_id == Shapes.PointMass:
            interaction_data.pointmass_pointmass(i, j, self.cutoff_radius)

        if ishape_id == Shapes.Sphere and jshape_id == Shapes.Sphere:
            interaction_data.sphere_sphere(i, j)

        if ishape_id == Shapes.Sphere and jshape_id == Shapes.Halfspace:
            interaction_data.sphere_halfspace(i, j)

        if ishape_id == Shapes.Sphere and jshape_id == Shapes.Box:
            interaction_data.sphere_box(i, j)
            
        if ishape_id == Shapes.Box and jshape_id == Shapes.Sphere:
            interaction_data.sphere_box(j, i, False)

        # Apply reductions for this i-j interaction:
        # -------------------------------------------------------------
        for app in self.apply_list[ishape*self.maxs + jshape]:
            app.add_reduction_variable()

        # The i-j block is executed only if the cutoff_condition of the i-j interaction is met
        self.sim.add_statement(Filter(self.sim, interaction_data.cutoff_condition, self.blocks[ishape*self.maxs + jshape]))
        self.apply_reductions(i, ishape, jshape)

    @pairs_device_block
    def lower(self):
        self.sim.module_name(f"{self.module_name}_local_interactions")
        if self.nbody == 2:
            neighbor_lists = None if self.use_cell_lists else self.sim.neighbor_lists
            for ishape in range(self.maxs):
                if self.include_shape(ishape):
                    # A kernel for each ishape
                    for i in ParticleFor(self.sim):
                        for _ in Filter(self.sim, ScalarOp.and_op(
                            ScalarOp.cmp(self.sim.particle_shape[i], self.sim.get_shape_id(ishape)),
                            ScalarOp.not_op(self.sim.particle_flags[i] & (Flags.Infinite | Flags.Global)))):
                            for jshape in range(self.maxs):
                                if self.include_interaction(ishape, jshape):
                                    # Inner loops for each jshaped neighbor
                                    for neigh in NeighborFor(self.sim, i, self.cell_lists, neighbor_lists, jshape):
                                        j = neigh.particle_index()
                                        self.compute_interaction(i, j, ishape, jshape)

        else:
            raise Exception("Interactions among more than two particles are currently not supported.")
