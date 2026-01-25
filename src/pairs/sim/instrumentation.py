from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Void
from pairs.ir.timers import Timers
from pairs.ir.types import Types
from pairs.sim.lowerable import FinalLowerable

class RegisterTimers(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        for t in range(Timers.Offset):
            Call_Void(self.sim, "::pairs::register_timer", [t, Timers.name(t)])

        # Interface modules
        for m in self.sim.interface_modules():
            if 'PairsSimulation' not in m.name and m.return_type==Types.Void:
                Call_Void(self.sim, "::pairs::register_timer", [m.module_id + Timers.Offset, "INTERFACE_MODULES::" + m.name])
        
        # Internal modules
        for m in self.sim.modules():
            Call_Void(self.sim, "::pairs::register_timer", [m.module_id + Timers.Offset, "INTERNAL_MODULES::" + m.name])


class RegisterMarkers(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        if self.sim._enable_profiler:
            # Only internal modules are profiled
            for m in self.sim.modules():
                if m.must_profile():
                    Call_Void(self.sim, "LIKWID_MARKER_REGISTER", [m.name])
