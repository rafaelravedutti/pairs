from pairs.code_gen.cgen import CGen
from pairs.sim.simulation import Simulation


def simulation(ref, dims=3, timesteps=100):
    return Simulation(CGen(f"{ref}.cpp"), dims, timesteps)
