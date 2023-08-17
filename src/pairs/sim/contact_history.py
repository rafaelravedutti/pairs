from pairs.ir.bin_op import BinOp
from pairs.ir.block import pairs_device_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import ParticleFor, For
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.interaction import NeighborFor
from pairs.sim.lowerable import Lowerable


class ContactHistory:
    def __init__(self, cell_lists):
        self.sim = cell_lists.sim
        self.cell_lists = cell_lists
        self.contact_lists = self.sim.add_array('contact_lists', [self.sim.particle_capacity, self.sim.neighbor_capacity], Types.Int32)
        self.num_contacts = self.sim.add_array('num_contacts', self.sim.particle_capacity, Types.Int32)


class BuildContactHistory(Lowerable):
    def __init__(self, sim, contact_history):
        super().__init__(sim)
        self.contact_history = contact_history

    @pairs_device_block
    def lower(self):
        contact_history = self.contact_history
        cell_lists = self.contact_history.cell_lists
        neighbor_lists = self.sim.neighbor_lists
        contact_lists = self.contact_history.contact_lists
        num_contacts = self.contact_history.num_contacts
        self.sim.module_name("build_contact_history")
        last_contact = self.sim.add_temp_var(0)
        contact_index = self.sim.add_temp_var(0)

        for i in ParticleFor(self.sim):
            last_contact.set(0)
            for neigh in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                j = neigh.particle_index()
                contact_index.set(-1)
                for k in For(self.sim, 0, num_contacts[i]):
                    for _ in Filter(self.sim, BinOp.cmp(contact_lists[i][k], j)):
                        contact_index.set(k)

                for _ in Filter(self.sim, BinOp.and_op(contact_index >= 0,
                                                       BinOp.neq(last_contact, contact_index))):
                    for contact_prop in self.sim.contact_properties:
                        if contact_prop.type() == Types.Vector:
                            for d in range(0, self.sim.ndims()):
                                tmp = self.sim.add_temp_var(contact_prop[i, last_contact][d])
                                contact_prop[i, last_contact][d].set(contact_prop[i, contact_index][d])
                                contact_prop[i, contact_index].set(tmp)

                        else:
                            tmp = self.sim.add_temp_var(contact_prop[i, last_contact])
                            contact_prop[i, last_contact].set(contact_prop[i, contact_index])
                            contact_prop[i, contact_index].set(tmp)

                    contact_lists[i][contact_index].set(contact_lists[i][last_contact])
                    contact_lists[i][last_contact].set(j)

                last_contact.set(last_contact + 1)

            last_contact.set(0)
            for neigh in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                j = neigh.particle_index()
                for _ in Filter(self.sim, BinOp.neq(contact_lists[i][last_contact], j)):
                    for contact_prop in self.sim.contact_properties:
                        if contact_prop.type() == Types.Vector:
                            for d in range(0, self.sim.ndims()):
                                contact_prop[i, last_contact][d].set(contact_prop.default()[d])
                        else:
                            contact_prop[i, last_contact].set(contact_prop.default())

                    contact_lists[i][last_contact].set(j)

                last_contact.set(last_contact + 1)
