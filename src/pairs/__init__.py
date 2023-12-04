from pairs.ir.types import Types
from pairs.code_gen.cgen import CGen
from pairs.code_gen.target import Target
from pairs.sim.domain_partitioners import DomainPartitioners
from pairs.sim.shapes import Shapes
from pairs.sim.simulation import Simulation


def simulation(ref, shapes, dims=3, timesteps=100, double_prec=False, use_contact_history=False, debug=False):
    return Simulation(CGen(ref, debug), shapes, dims, timesteps, double_prec, use_contact_history)

def target_cpu():
    return Target(Target.Backend_CPP, Target.Feature_CPU)

def target_gpu():
    return Target(Target.Backend_CUDA, Target.Feature_GPU)

def int32():
    return Types.Int32

def float():
    return Types.Float

def double():
    return Types.Double

def real():
    return Types.Real

def vector():
    return Types.Vector

def matrix():
    return Types.Matrix

def quaternion():
    return Types.Quaternion

def point_mass():
    return Shapes.PointMass

def sphere():
    return Shapes.Sphere

def halfspace():
    return Shapes.Halfspace

def regular_domain_partitioner():
    return DomainPartitioners.Regular

def regular_domain_partitioner_xy():
    return DomainPartitioners.RegularXY
