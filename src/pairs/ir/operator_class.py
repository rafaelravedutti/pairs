from pairs.ir.matrices import MatrixOp
from pairs.ir.quaternions import QuaternionOp
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types
from pairs.ir.vectors import VectorOp


class OperatorClass:
    def from_type(t):
        return VectorOp if t == Types.Vector else \
               MatrixOp if t == Types.Matrix else \
               QuaternionOp if t == Types.Quaternion else \
               ScalarOp

    def from_type_list(type_list):
        if Types.Quaternion in type_list:
            return OperatorClass.from_type(Types.Quaternion)

        if Types.Matrix in type_list:
            return OperatorClass.from_type(Types.Matrix)

        if Types.Vector in type_list:
            return OperatorClass.from_type(Types.Vector)

        return OperatorClass.from_type(type_list[0])
