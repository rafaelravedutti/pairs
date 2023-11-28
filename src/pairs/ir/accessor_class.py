from pairs.ir.matrices import MatrixAccess
from pairs.ir.quaternions import QuaternionAccess
from pairs.ir.types import Types
from pairs.ir.vectors import VectorAccess


class AccessorClass:
    def from_type(t):
        return VectorAccess if t == Types.Vector else \
               MatrixAccess if t == Types.Matrix else \
               QuaternionAccess if t == Types.Quaternion else \
               None
