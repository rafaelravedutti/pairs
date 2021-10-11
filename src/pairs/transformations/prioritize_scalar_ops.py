from pairs.ir.bin_op import BinOp
from pairs.ir.data_types import Type_Float, Type_Vector
from pairs.ir.mutator import Mutator


class PrioritazeScalarOps(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def can_rearrange(op1, op2):
        return op1 == op2 and op1 in ['+', '*']

    def mutate_BinOp(self, ast_node):
        sim = ast_node.sim
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)

        if ast_node.type() == Type_Vector:
            lhs = ast_node.lhs
            rhs = ast_node.rhs
            op = ast_node.op

            if( isinstance(lhs, BinOp) and lhs.type() == Type_Vector and rhs.type() == Type_Float and \
                PrioritazeScalarOps.can_rearrange(op, lhs.op) ):

                if lhs.lhs.type() == Type_Vector and lhs.rhs.type() == Type_Float:
                    ast_node.reassign(lhs.lhs, BinOp(sim, lhs.rhs, rhs, op), op)
                    #return BinOp(sim, lhs.lhs, BinOp(sim, lhs.rhs, rhs, op), op)

                if lhs.rhs.type() == Type_Vector and lhs.lhs.type() == Type_Float:
                    ast_node.reassign(lhs.rhs, BinOp(sim, lhs.lhs, rhs, op), op)
                    #return BinOp(sim, lhs.rhs, BinOp(sim, lhs.lhs, rhs, op), op)

            if( isinstance(rhs, BinOp) and rhs.type() == Type_Vector and lhs.type() == Type_Float and \
                PrioritazeScalarOps.can_rearrange(op, rhs.op) ):

                if rhs.lhs.type() == Type_Vector and rhs.rhs.type() == Type_Float:
                    ast_node.reassign(rhs.lhs, BinOp(sim, rhs.rhs, lhs, op), op)
                    #return BinOp(sim, rhs.lhs, BinOp(sim, rhs.rhs, lhs, op), op)

                if rhs.rhs.type() == Type_Vector and rhs.lhs.type() == Type_Float:
                    ast_node.reassign(rhs.rhs, BinOp(sim, rhs.lhs, lhs, op), op)
                    #return BinOp(sim, rhs.rhs, BinOp(sim, rhs.lhs, lhs, op), op)

        return ast_node


def prioritaze_scalar_ops(ast_node):
    prioritaze = PrioritazeScalarOps(ast_node)
    prioritaze.mutate()
