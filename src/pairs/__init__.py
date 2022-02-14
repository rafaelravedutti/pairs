from pairs.code_gen.cgen import CGen
from pairs.sim.simulation import Simulation


def simulation(ref, dims=3, timesteps=100, debug=False):
    return Simulation(CGen(f"{ref}.cpp", debug), dims, timesteps)
