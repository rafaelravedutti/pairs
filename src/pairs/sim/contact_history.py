from pairs.ir.block import pairs_device_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import ParticleFor
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.interaction import ParticleInteraction
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
        neighbor_lists = self.contact_history.neighbor_lists
        contact_lists = self.contact_history.contact_lists
        num_contacts = self.contact_history.num_contacts
        sim.module_name("build_contact_history")
        last_contact_id = self.sim.add_temp_var(0)
        neighbor_contact = self.sim.add_temp_var(0)

        for i in ParticleFor(self.sim):
            last_contact_id.set(0)
            for j in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                neighbor_contact.set(-1)
                for k in For(self.sim, 0, num_contacts[i]):
                    if contact_lists[i][k] == j:
                        neighbor_contact.set(k)

                for _ in Filter(self.sim, neighbor_contact >= 0):
                    contact_lists[i][k].set(contact_lists[i][last_contact_id])

                    for contact_prop in self.sim.contact_properties():
                        tmp = self.sim.add_temp_var(contact_prop[i][last_contact_id])
                        contact_prop[i][last_contact_id].set(contact_prop[i][k])
                        contact_prop[i][k].set(tmp)

                    contact_lists[i][last_contact_id].set(j)

                last_contact_id.set(last_contact_id + 1)

            last_contact_id.set(0)
            for j in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                neighbor_contact.set(-1)
                for k in For(self.sim, 0, num_contacts[i]):
                    if contact_lists[i][k] == j:
                        neighbor_contact.set(k)

                for _ in Filter(self.sim, neighbor_contact < 0):
                    for contact_prop in self.sim.contact_properties():
                        contact_prop[i][last_contact_id].set(contact_prop.default())

                    contact_lists[i][last_contact_id].set(j)

                last_contact_id.set(last_contact_id + 1)
