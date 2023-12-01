from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Void
from pairs.ir.timers import Timers
from pairs.sim.lowerable import FinalLowerable

class RegisterTimers(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        for t in range(Timers.Offset):
            Call_Void(self.sim, "pairs::register_timer", [t, Timers.name(t)])

        for m in self.sim.module_list:
            if m.name != 'main':
                Call_Void(self.sim, "pairs::register_timer", [m.module_id + Timers.Offset, m.name])


class RegisterMarkers(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        if self.sim._enable_profiler:
            for m in self.sim.module_list:
                if m.name != 'main' and m.must_profile():
                    Call_Void(self.sim, "LIKWID_MARKER_REGISTER", [m.name])
