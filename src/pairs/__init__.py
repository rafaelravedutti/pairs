from pairs.code_gen.cgen import CGen
from pairs.sim.particle_simulation import ParticleSimulation


def simulation(ref, dims=3, timesteps=100):
    return ParticleSimulation(CGen(f"{ref}.cpp"), dims, timesteps)
