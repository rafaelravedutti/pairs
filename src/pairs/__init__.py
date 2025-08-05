from pairs.ir.types import Types
from pairs.ir.sync_modes import SyncModes
from pairs.code_gen.cgen import CGen
from pairs.code_gen.target import Target
from pairs.sim.domain_partitioners import DomainPartitioners
from pairs.sim.shapes import Sphere, Halfspace, PointMass, Box, Clump
from pairs.sim.simulation import Simulation


def simulation(
    ref,
    dims=3,
    double_prec=False,
    use_contact_history=False,
    particle_capacity=800000,
    neighbor_capacity=20,
    debug=False):

    return Simulation(
        CGen(ref, debug), 
        dims, 
        double_prec, 
        use_contact_history, 
        particle_capacity, 
        neighbor_capacity)


# ======================================================================
# Types
# ======================================================================

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

# ======================================================================
# SyncModes
# ======================================================================

def always():
    return SyncModes.ALWAYS

def never():
    return SyncModes.NEVER

def on_reneighbor():
    return SyncModes.ON_RENEIGHBOR

def on_reduction():
    return SyncModes.ON_REDUCTION

# ======================================================================
# Shapes
# ======================================================================

def point_mass():
    return PointMass()

def sphere(radius):
    return Sphere(radius)

def halfspace(normal):
    return Halfspace(normal)

def box(edge_length, rotation_matrix):
    return Box(edge_length, rotation_matrix)

def clump(local_positions, local_radii, rotation_matrix):
    return Clump(local_positions, local_radii, rotation_matrix)

# ======================================================================
# DomainPartitioners
# ======================================================================

def regular_domain_partitioner():
    return DomainPartitioners.Regular

def regular_domain_partitioner_xy():
    return DomainPartitioners.RegularXY

def block_forest():
    return DomainPartitioners.BlockForest
