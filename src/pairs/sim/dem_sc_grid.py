from pairs.ir.assign import Assign
from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Int
from pairs.sim.lowerable import Lowerable


class DEMSCGrid(Lowerable):
    def __init__(
        self, sim, xmax, ymax, zmax, spacing, diameter, min_diameter, max_diameter, initial_velocity, particle_density, ntypes):

        super().__init__(sim)
        self._xmax = xmax
        self._ymax = ymax
        self._zmax = zmax
        self._spacing = spacing
        self._diameter = diameter
        self._min_diameter = min_diameter
        self._max_diameter = max_diameter
        self._initial_velocity = initial_velocity
        self._particle_density = particle_density
        self._ntypes = ntypes
        #sim.set_domain([0.0, 0.0, 0.0, self._xprd, self._yprd, self._zprd])

    @pairs_inline
    def lower(self):
        Assign(self.sim, self.sim.nlocal,
            Call_Int(self.sim, "pairs::dem_sc_grid",
                [self._xmax, self._ymax, self._zmax, self._spacing, self._diameter, self._min_diameter, self._max_diameter,
                 self._initial_velocity, self._particle_density, self._ntypes]))
