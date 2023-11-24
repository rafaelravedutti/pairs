from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Void
from pairs.sim.lowerable import FinalLowerable


class RegisterTimers(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        Call_Void(self.sim, "pairs::register_timer", [0, "all"])

        for m in self.sim.module_list:
            if m.name != 'main':
                Call_Void(self.sim, "pairs::register_timer", [m.module_id + 1, m.name])


class RegisterMarkers(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        if self.sim._enable_profiler:
            for m in self.sim.module_list:
                if m.name != 'main' and m.must_profile():
                    Call_Void(self.sim, "LIKWID_MARKER_REGISTER", [m.name])
