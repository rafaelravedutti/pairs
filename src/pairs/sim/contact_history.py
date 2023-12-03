from pairs.ir.assign import Assign
from pairs.ir.block import pairs_device_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import ParticleFor, For, While
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.interaction import NeighborFor
from pairs.sim.lowerable import Lowerable


class ContactHistory:
    def __init__(self, sim):
        self.sim = sim
        self.contact_lists = self.sim.add_array('contact_lists', [self.sim.particle_capacity, self.sim.neighbor_capacity], Types.Int32)
        self.contact_used = self.sim.add_array('contact_used', [self.sim.particle_capacity, self.sim.neighbor_capacity], Types.Int32)
        self.num_contacts = self.sim.add_array('num_contacts', self.sim.particle_capacity, Types.Int32)


class BuildContactHistory(Lowerable):
    def __init__(self, sim, contact_history, cell_lists):
        super().__init__(sim)
        self.contact_history = contact_history
        self.cell_lists = cell_lists

    @pairs_device_block
    def lower(self):
        contact_history = self.contact_history
        cell_lists = self.cell_lists
        particle_uid = self.sim.particle_uid
        neighbor_lists = self.sim.neighbor_lists
        contact_lists = self.contact_history.contact_lists
        contact_used = self.contact_history.contact_used
        num_contacts = self.contact_history.num_contacts
        self.sim.module_name("build_contact_history")

        for i in ParticleFor(self.sim):
            last_contact = self.sim.add_temp_var(0)
            contact_index = self.sim.add_temp_var(0)

            for neigh in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                j = neigh.particle_index()
                Assign(self.sim, contact_index, -1)
                for k in For(self.sim, 0, num_contacts[i]):
                    for _ in Filter(self.sim, ScalarOp.cmp(contact_lists[i][k], particle_uid[j])):
                        Assign(self.sim, contact_index, k)

                for _ in Filter(self.sim,
                                ScalarOp.and_op(contact_index >= 0,
                                                ScalarOp.neq(last_contact, contact_index))):
                    for contact_prop in self.sim.contact_properties:
                        tmp = self.sim.add_temp_var(contact_prop[i, last_contact])
                        Assign(self.sim, contact_prop[i, last_contact], contact_prop[i, contact_index])
                        Assign(self.sim, contact_prop[i, contact_index], tmp)

                    Assign(self.sim, contact_lists[i][contact_index], contact_lists[i][last_contact])
                    Assign(self.sim, contact_lists[i][last_contact], particle_uid[j])

                Assign(self.sim, last_contact, last_contact + 1)

            Assign(self.sim, last_contact, 0)
            for neigh in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                j = neigh.particle_index()
                for _ in Filter(self.sim, ScalarOp.neq(contact_lists[i][last_contact], particle_uid[j])):
                    for contact_prop in self.sim.contact_properties:
                        Assign(self.sim, contact_prop[i, last_contact], contact_prop.default())

                    Assign(self.sim, contact_lists[i][last_contact], particle_uid[j])

                Assign(self.sim, last_contact, last_contact + 1)

            Assign(self.sim, num_contacts[i], last_contact)


class ResetContactHistoryUsageStatus(Lowerable):
    def __init__(self, sim, contact_history):
        super().__init__(sim)
        self.contact_history = contact_history

    @pairs_device_block
    def lower(self):
        contact_used = self.contact_history.contact_used
        num_contacts = self.contact_history.num_contacts
        self.sim.module_name("reset_contact_history_usage_status")

        for i in ParticleFor(self.sim):
            for c in For(self.sim, 0, num_contacts[i]):
                Assign(self.sim, contact_used[i][c], 0)


class ClearUnusedContactHistory(Lowerable):
    def __init__(self, sim, contact_history):
        super().__init__(sim)
        self.contact_history = contact_history

    @pairs_device_block
    def lower(self):
        contact_lists = self.contact_history.contact_lists
        contact_used = self.contact_history.contact_used
        num_contacts = self.contact_history.num_contacts
        self.sim.module_name("clear_unused_contact_history")

        if self.sim.neighbor_lists is None:
            for i in ParticleFor(self.sim):
                c = self.sim.add_temp_var(0)
                for _ in While(self.sim, c < num_contacts[i]):
                    for unused in Branch(self.sim, ScalarOp.cmp(contact_used[i][c], 0)):
                        if unused:
                            last_contact = num_contacts[i] - 1
                            for _ in Filter(self.sim, last_contact > 0):
                                for contact_prop in self.sim.contact_properties:
                                    Assign(self.sim, contact_prop[i, c], contact_prop[i, last_contact])

                                Assign(self.sim, contact_lists[i][c], contact_lists[i][last_contact])
                                Assign(self.sim, contact_used[i][c], contact_used[i][last_contact])

                            Assign(self.sim, num_contacts[i], num_contacts[i] - 1)

                        else:
                            Assign(self.sim, c, c + 1)

        else:
            for i in ParticleFor(self.sim):
                for c in For(self.sim, 0, num_contacts[i]):
                    for _ in Filter(self.sim, ScalarOp.cmp(contact_used[i][c], 0)):
                        for contact_prop in self.sim.contact_properties:
                            Assign(self.sim, contact_prop[i, c], contact_prop.default())
