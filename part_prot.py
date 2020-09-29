from code_gen.cgen import CGen
from sim.particle_simulation import ParticleSimulation

def simulation(dims=3, timesteps=100):
    return ParticleSimulation(CGen, dims, timesteps)
