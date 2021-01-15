from code_gen.cgen import CGen
from sim.particle_simulation import ParticleSimulation


def simulation(ref, dims=3, timesteps=100):
    return ParticleSimulation(CGen(f"{ref}.c"), dims, timesteps)
