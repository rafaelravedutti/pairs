from pairs.ir.assign import Assign
from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Int, Call_Void
from pairs.ir.particle_attributes import ParticleAttributeList
from pairs.ir.types import Types
from pairs.sim.grid import Grid3D
from pairs.sim.lowerable import Lowerable


class CopperFCCLattice(Lowerable):
    def __init__(self, sim, nx, ny, nz, rho, temperature, ntypes):
        super().__init__(sim)
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._rho = rho
        self._temperature = temperature
        self._ntypes = ntypes
        lattice = pow((4.0 / rho), (1.0 / 3.0))
        self._xprd = nx * lattice
        self._yprd = ny * lattice
        self._zprd = nz * lattice
        sim.set_domain([0.0, 0.0, 0.0, self._xprd, self._yprd, self._zprd])

    @pairs_inline
    def lower(self):
        Assign(self.sim, self.sim.nlocal,
            Call_Int(self.sim, "pairs::copper_fcc_lattice",
                [self._nx, self._ny, self._nz,
                 self._xprd, self._yprd, self._zprd,
                 self._rho, self._ntypes]))

        Call_Void(self.sim, "pairs::adjust_thermo",
            [self.sim.nlocal, self._xprd, self._yprd, self._zprd, self._temperature])
