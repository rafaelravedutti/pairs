from pairs.ir.bin_op import BinOp
from pairs.ir.mutator import Mutator
from pairs.ir.types import Types


class PrioritizeScalarOps(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def can_rearrange(op1, op2):
        return op1 == op2 and op1 in ['+', '*']

    def mutate_BinOp(self, ast_node):
        sim = ast_node.sim
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)

        if ast_node.type() == Types.Vector:
            lhs = ast_node.lhs
            rhs = ast_node.rhs
            op = ast_node.op

            if( isinstance(lhs, BinOp) and lhs.type() == Types.Vector and Types.is_real(rhs.type()) and \
                PrioritizeScalarOps.can_rearrange(op, lhs.op) ):

                if lhs.lhs.type() == Types.Vector and Types.is_real(lhs.rhs.type()):
                    ast_node.reassign(lhs.lhs, BinOp(sim, lhs.rhs, rhs, op), op)
                    #return BinOp(sim, lhs.lhs, BinOp(sim, lhs.rhs, rhs, op), op)

                if lhs.rhs.type() == Types.Vector and Types.is_real(lhs.lhs.type()):
                    ast_node.reassign(lhs.rhs, BinOp(sim, lhs.lhs, rhs, op), op)
                    #return BinOp(sim, lhs.rhs, BinOp(sim, lhs.lhs, rhs, op), op)

            if( isinstance(rhs, BinOp) and rhs.type() == Types.Vector and Types.is_real(lhs.type()) and \
                PrioritizeScalarOps.can_rearrange(op, rhs.op) ):

                if rhs.lhs.type() == Types.Vector and Types.is_real(rhs.rhs.type()):
                    ast_node.reassign(rhs.lhs, BinOp(sim, rhs.rhs, lhs, op), op)
                    #return BinOp(sim, rhs.lhs, BinOp(sim, rhs.rhs, lhs, op), op)

                if rhs.rhs.type() == Types.Vector and Types.is_real(rhs.lhs.type()):
                    ast_node.reassign(rhs.rhs, BinOp(sim, rhs.lhs, lhs, op), op)
                    #return BinOp(sim, rhs.rhs, BinOp(sim, rhs.lhs, lhs, op), op)

        return ast_node


def prioritize_scalar_ops(ast_node):
    prioritize = PrioritizeScalarOps(ast_node)
    prioritize.mutate()
