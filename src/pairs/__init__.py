from pairs.code_gen.cgen import CGen
from pairs.code_gen.target import Target
from pairs.sim.simulation import Simulation


def simulation(ref, dims=3, timesteps=100, debug=False):
    return Simulation(CGen(ref, debug), dims, timesteps)

def target_cpu():
    return Target(Target.Backend_CPP, Target.Feature_CPU)

def target_gpu():
    return Target(Target.Backend_CUDA, Target.Feature_GPU)
