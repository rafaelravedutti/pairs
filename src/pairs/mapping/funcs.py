import ast
import inspect
from pairs.ir.assign import Assign
from pairs.ir.branches import Branch, Filter
from pairs.ir.lit import Lit
from pairs.ir.loops import For, ParticleFor
from pairs.ir.operators import Operators
from pairs.ir.operator_class import OperatorClass
from pairs.ir.properties import ContactProperty
from pairs.ir.parameters import Parameter
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types
from pairs.mapping.keywords import Keywords
from pairs.sim.flags import Flags
from pairs.sim.interaction import ParticleInteraction
from pairs.sim.global_interaction import  GlobalLocalInteraction, GlobalGlobalInteraction, GlobalReduction, SortGlobals, PackGlobals, ResetReductionProps, ReduceGlobals, UnpackGlobals
from pairs.ir.module import Module
from pairs.ir.block import Block, pairs_block
from pairs.sim.lowerable import Lowerable

class UndefinedSymbol():
    def __init__(self, symbol_id):
        self.symbol_id = symbol_id

    def type(self):
        return Types.Invalid


class FetchParticleFuncInfo(ast.NodeVisitor):
    def __init__(self):
        self._params = []

    def visit_arg(self, node):
        self._params.append(node.arg)

    def nparams(self):
        return len(self._params)

    def params(self):
        return self._params


class BuildParticleIR(ast.NodeVisitor):
    def get_unary_op(op):
        unary_ops = {
            ast.UAdd: Operators.UAdd,
            ast.USub: Operators.USub,
            ast.Invert: Operators.Invert,
            ast.Not: Operators.Not
        }

        op_type = type(op)
        if op_type in unary_ops:
            return unary_ops[op_type]

        raise Exception("Invalid operator: {}".format(ast.dump(op)))

    def get_binary_op(op):
        binary_ops = {
            ast.Add: Operators.Add,
            ast.Sub: Operators.Sub,
            ast.Mult: Operators.Mul,
            ast.Div: Operators.Div,
            #ast.FloorDiv: Operators.FloorDiv,
            ast.Mod: Operators.Mod,
            ast.Eq: Operators.Eq,
            ast.NotEq: Operators.Neq,
            ast.Gt: Operators.Gt,
            ast.Lt: Operators.Lt,
            ast.GtE: Operators.Geq,
            ast.LtE: Operators.Leq,
            ast.BitAnd: Operators.BinAnd,
            ast.BitOr: Operators.BinOr,
            ast.BitXor: Operators.BinXor,
            #ast.LShift: Operators.LShift,
            #ast.RShift: Operators.RShift,
            ast.And: Operators.And,
            ast.Or: Operators.Or
        }

        op_type = type(op)
        if op_type in binary_ops:
            return binary_ops[op_type]

        raise Exception("Invalid operator: {}".format(ast.dump(op)))

    def __init__(self, sim, ctx_symbols={}, func_params={}):
        self.sim = sim
        self.ctx_symbols = ctx_symbols.copy()
        self.func_params = func_params.copy()
        self.keywords = Keywords(sim)

    def add_symbols(self, symbols):
        self.ctx_symbols.update(symbols)

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Only one target is allowed on assignments!"
        lhs = self.visit(node.targets[0])
        rhs = self.visit(node.value)

        if isinstance(lhs, UndefinedSymbol):
            self.add_symbols({lhs.symbol_id: rhs})
            rhs.set_label(lhs.symbol_id)
        else:
            Assign(self.sim, lhs, rhs)

    def visit_AugAssign(self, node):
        lhs = self.visit(node.target)
        # We need a copy of the target object so it is properly visited during
        # compiler analyses and transformations
        lhs_copy = self.visit(node.target)
        rhs = self.visit(node.value)
        op_class = OperatorClass.from_type_list([lhs.type(), rhs.type()])
        bin_op = op_class(self.sim, lhs_copy, rhs, BuildParticleIR.get_binary_op(node.op))

        assert not isinstance(lhs, UndefinedSymbol), \
            f"Invalid AugAssign: symbol {lhs} not defined yet!"
        Assign(self.sim, lhs, bin_op)

    def visit_BinOp(self, node):
        #print(ast.dump(node))
        lhs = self.visit(node.left)
        assert not isinstance(lhs, UndefinedSymbol), f"Undefined lhs used in BinOp: {lhs.symbol_id}"
        rhs = self.visit(node.right)
        assert not isinstance(rhs, UndefinedSymbol), f"Undefined rhs used in BinOp: {rhs.symbol_id}"
        operator = BuildParticleIR.get_binary_op(node.op)

        if operator == Operators.Mul:
            if Types.Matrix in (lhs.type(), rhs.type()):
                return self.keywords.keyword_matrix_multiplication([lhs, rhs])

            if Types.Quaternion in (lhs.type(), rhs.type()):
                return self.keywords.keyword_quaternion_multiplication([lhs, rhs])

        op_class = OperatorClass.from_type_list([lhs.type(), rhs.type()])
        return op_class(self.sim, lhs, rhs, BuildParticleIR.get_binary_op(node.op))

    def visit_BoolOp(self, node):
        #print(ast.dump(node))
        op = BuildParticleIR.get_binary_op(node.op)
        first = self.visit(node.values[0])
        assert not isinstance(first, UndefinedSymbol), \
            f"Undefined operator used in BoolOp: {first.symbol_id}"

        expr = first
        for value in node.values[1:]:
            voper = self.visit(value)
            assert not isinstance(voper, UndefinedSymbol), \
                f"Undefined operator used in BoolOp: {voper.symbol_id}"
            expr = ScalarOp(self.sim, expr, voper, op)

        return expr

    def visit_Call(self, node):
        func = self.visit(node.func).symbol_id
        args = [self.visit(a) for a in node.args]

        if self.keywords.exists(func):
            if func == 'apply':
                args += [self.ctx_symbols['__i__'], self.ctx_symbols['__j__']]

            return self.keywords(func, args)

        if func in ['delta', 'squared_distance', 'penetration_depth', 'contact_point', 'contact_normal']:
            return self.ctx_symbols[f"__{func}__"]

        raise Exception(f"Undefined function called: {func}")

    def visit_Constant(self, node):
        return Lit(self.sim, node.value)

    def visit_Compare(self, node):
        #print(ast.dump(node))
        valid_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)
        if len(node.ops) != 1 or not isinstance(node.ops[0], valid_ops):
            raise Exception(f"Chained comparisons or unsupported comparison found: {ast.dump(node)}")

        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        operator = BuildParticleIR.get_binary_op(node.ops[0])
        op_class = OperatorClass.from_type_list([lhs.type(), rhs.type()])
        return op_class(self.sim, lhs, rhs, operator)

    def visit_If(self, node):
        condition = self.visit(node.test)
        one_way = node.orelse is None or len(node.orelse) == 0

        if one_way:
            for _ in Filter(self.sim, condition):
                for stmt in node.body:
                    self.visit(stmt)

        else:
            for test in Branch(self.sim, condition):
                if test:
                    for stmt in node.body:
                        self.visit(stmt)
                else:
                    for stmt in node.orelse:
                        self.visit(stmt)

    def visit_IfExp(self, node):
        condition = self.visit(node.test)
        for test in Branch(self.sim, condition):
            if test:
                for stmt in node.body:
                    self.visit(stmt)
            else:
                for stmt in node.orelse:
                    self.visit(stmt)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        symbol_types = [
            self.ctx_symbols.get,
            self.func_params.get,
            self.sim.array,
            self.sim.property,
            self.sim.feature_property,
            self.sim.contact_property,
            self.sim.var
        ]
        
        for symbol_func in symbol_types:
            result = symbol_func(node.id)
            if result is not None:
                return result

        return UndefinedSymbol(node.id)

    def visit_Num(self, node):
        return Lit(self.sim, node.n)

    def visit_Subscript(self, node):
        #print(ast.dump(node))
        value = self.visit(node.value)
        index = self.visit(node.slice)

        if isinstance(value, ContactProperty):
            i = index[0]
            j = index[1]
            contact_used = self.sim._contact_history.contact_used

            if self.sim.neighbor_lists is None:
                if '__contact_id__' not in self.ctx_symbols:
                    particle_uid = self.sim.particle_uid
                    contact_lists = self.sim._contact_history.contact_lists
                    num_contacts = self.sim._contact_history.num_contacts
                    contact_id = self.sim.add_temp_var(-1)

                    for k in For(self.sim, 0, num_contacts[i]):
                        for _ in Filter(self.sim, ScalarOp.cmp(contact_lists[i][k], particle_uid[j])):
                            Assign(self.sim, contact_id, k)

                    for _ in Filter(self.sim, ScalarOp.cmp(contact_id, -1)):
                        last_contact = num_contacts[i]
                        Assign(self.sim, contact_id, last_contact)
                        Assign(self.sim, num_contacts[i], last_contact + 1)
                        Assign(self.sim, contact_lists[i][last_contact], particle_uid[j])

                        for contact_prop in self.sim.contact_properties:
                            Assign(self.sim, contact_prop[i, last_contact], contact_prop.default())

                    Assign(self.sim, contact_used[i][contact_id], 1)
                    self.ctx_symbols.update({'__contact_id__': contact_id})

                return value[i, self.ctx_symbols['__contact_id__']]

            else:
                if '__contact_id__' not in self.ctx_symbols:
                    Assign(self.sim, contact_used[i][j], 1)
                    self.ctx_symbols.update({'__contact_id__': j})

        return value[index]

    def visit_Tuple(self, node):
        #print(ast.dump(node))
        return tuple(self.visit(v) for v in node.elts)

    def visit_UnaryOp(self, node):
        #print(ast.dump(node))
        operand = self.visit(node.operand)
        assert not isinstance(operand, UndefinedSymbol), \
            f"Undefined operand used in UnaryOp: {operand.symbol_id}"
        op_class = OperatorClass.from_type(operand.type())
        return op_class(self.sim, operand, None, BuildParticleIR.get_unary_op(node.op))

class OneBodyKernel(Lowerable):
    def __init__(self, sim, module_name, run_on_device, profile):
        super().__init__(sim)
        self.block = Block(sim, [])
        self.module_name = module_name
        self.run_on_device = run_on_device
        self.profile = profile

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)
        for i in ParticleFor(self.sim):
            yield i
        self.sim.leave()
    
    @pairs_block
    def lower(self):
        self.sim.module_name(self.module_name)
        self.sim.add_statement(self.block)

def compute(sim, func, cutoff_radius=None, symbols={}, parameters={}, compute_globals=False, run_on_device=True, profile=False):
    src = inspect.getsource(func)
    tree = ast.parse(src, mode='exec')
    #print(ast.dump(ast.parse(src, mode='exec')))

    # Fetch function info
    info = FetchParticleFuncInfo()
    info.visit(tree)
    params = info.params()
    nparams = info.nparams()

    # Compute functions must have parameters
    assert nparams > 0, "Number of parameters from compute functions must be higher than zero!"

    # Convert literal symbols
    symbols = {symbol: Lit.cvt(sim, value) for symbol, value in symbols.items()}
    parameters = {pname: Parameter(sim, pname, ptype) for pname, ptype in parameters.items()}

    sim.init_block()
    sim.module_name(func.__name__)

    if nparams == 1:
        for i in OneBodyKernel(sim, func.__name__, run_on_device=run_on_device, profile=profile):
            ir = BuildParticleIR(sim, symbols, parameters)
            ir.add_symbols({params[0]: i})
            ir.visit(tree)

    else:
        # Compute local-local and local-global interactions
        particle_interaction = ParticleInteraction(sim, func.__name__, nparams, cutoff_radius, run_on_device=run_on_device, profile=profile)
        for interaction_data in particle_interaction:
            # Start building IR
            ir = BuildParticleIR(sim, symbols, parameters)
            ir.add_symbols({
                params[0]: interaction_data.i(),
                params[1]: interaction_data.j(),
                '__i__': interaction_data.i(),
                '__j__': interaction_data.j(),
                '__delta__': interaction_data.delta(),
                '__squared_distance__': interaction_data.squared_distance(),
                '__penetration_depth__': interaction_data.penetration_depth(),
                '__contact_point__': interaction_data.contact_point(),
                '__contact_normal__': interaction_data.contact_normal()
            })
            ir.visit(tree)

        if compute_globals:
            # If compute_globals is enabled, global interactions and reductions become 
            # part of the same module as local interactions
            global_reduction = GlobalReduction(sim, func.__name__, particle_interaction)
            
            SortGlobals(global_reduction)           # Sort global bodies with respect to their uid
            PackGlobals(global_reduction)           # Pack reduction properties in uid order in an intermediate buffer
            ResetReductionProps(global_reduction)   # Reset reduction properties to be prepared for local reduction 
            
            # Compute local contirbutions on global bodies
            global_local_interactions = GlobalLocalInteraction(sim, func.__name__, nparams)
            for interaction_data in global_local_interactions:
                ir = BuildParticleIR(sim, symbols, parameters)
                ir.add_symbols({
                    params[0]: interaction_data.i(),
                    params[1]: interaction_data.j(),
                    '__i__': interaction_data.i(),
                    '__j__': interaction_data.j(),
                    '__delta__': interaction_data.delta(),
                    '__squared_distance__': interaction_data.squared_distance(),
                    '__penetration_depth__': interaction_data.penetration_depth(),
                    '__contact_point__': interaction_data.contact_point(),
                    '__contact_normal__': interaction_data.contact_normal()
                })
                ir.visit(tree)


            PackGlobals(global_reduction, False)    # Pack local contributions in reduction buffer in uid order
            ReduceGlobals(global_reduction)         # Inplace reduce local contributions over global bodies in the reduction buffer
            UnpackGlobals(global_reduction)         # Add the reduced properties with the intermediate buffer and unpack into global bodies 

            # Compute global-global interactions
            global_global_interactions = GlobalGlobalInteraction(sim, func.__name__, nparams)
            for interaction_data in global_global_interactions:
                ir = BuildParticleIR(sim, symbols, parameters)
                ir.add_symbols({
                    params[0]: interaction_data.i(),
                    params[1]: interaction_data.j(),
                    '__i__': interaction_data.i(),
                    '__j__': interaction_data.j(),
                    '__delta__': interaction_data.delta(),
                    '__squared_distance__': interaction_data.squared_distance(),
                    '__penetration_depth__': interaction_data.penetration_depth(),
                    '__contact_point__': interaction_data.contact_point(),
                    '__contact_normal__': interaction_data.contact_normal()
                })
                ir.visit(tree)

            
            
    # User defined functions are wrapped inside seperate interface modules here.
    # The udf's have the same name as their interface module but they get implemented in the pairs::internal scope.
    sim.build_interface_module_with_statements()  
    
