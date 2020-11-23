from ast.block import Block
from ast.branches import Branch
from ast.data_types import Type_Float, Type_Vector
from ast.math import Sqrt
from ast.select import Select
import clang.cindex
from clang.cindex import CursorKind as kind


def print_tree(node, indent=0):
    if node is not None:
        spaces = ' ' * indent
        line = node.location.line
        column = node.location.column
        print(f"{spaces}{line}:{column}> {node.spelling} ({node.kind})")

        for child in node.get_children():
            print_tree(child, indent + 2)

def get_subtree(node, ref):
    splitted_ref = ref.split("::", 1)
    if len(splitted_ref) == 2:
        look_for, remaining = splitted_ref
    else:
        look_for = splitted_ref[0]
        remaining = None

    for child in node.get_children():
        cond_namespace = remaining is not None and \
                         (child.kind == kind.NAMESPACE or
                          child.kind == kind.CLASS_DECL) and \
                         child.spelling == look_for

        cond_func = remaining is None and \
                    (child.kind == kind.FUNCTION_TEMPLATE or \
                     child.kind == kind.CXX_METHOD or \
                     child.kind == kind.NAMESPACE or \
                     child.kind == kind.CLASS_DECL) and \
                    child.spelling == look_for

        if cond_namespace or cond_func:
            if remaining is None:
                return child

            subtree_res = get_subtree(child, remaining)
            if subtree_res is not None:
                return subtree_res

    return None

def get_class_method(node, class_ref, function_name):
    class_ref_ = class_ref if class_ref.startswith("class ") \
                 else "class " + class_ref

    for child in node.get_children():
        if child.spelling == function_name and \
           (child.kind == kind.CXX_METHOD or \
            child.kind == kind.FUNCTION_TEMPLATE):
            for grandchild in child.get_children():
                if grandchild.kind == kind.TYPE_REF and \
                   grandchild.spelling == class_ref_:
                    return child

        child_res = get_class_method(child, class_ref, function_name)
        if child_res is not None:
            return child_res

    return None

def getVelocityAtWFPoint(sim, params):
    p_idx = params[0]
    #ac    = params[1]
    wf_pt = params[2]
    lin_vel = sim.property('velocity')
    ang_vel = sim.property('angular_velocity')
    position = sim.property('position')
    return lin_vel[p_idx] + ang_vel[p_idx] * (wf_pt - position[p_idx])

def addForceAtWFPosAtomic(sim, params):
    p_idx = params[0]
    #ac    = params[1]
    f     = params[2]
    wf_pt = params[3]
    force = sim.property('force')
    torque = sim.property('torque')
    position = sim.property('position')
    force[p_idx].add(f)
    torque[p_idx].add((wf_pt - position[p_idx]) * f)

def getType(sim, params):
    return sim.property('type')[params[0]]

def getStiffness(sim, params):
    type_a = params[0]
    type_b = params[1]
    ntypes = sim.var('ntypes')
    return sim.array('stiffness')[type_a * ntypes + type_b]

def getDampingN(sim, params):
    type_a = params[0]
    type_b = params[1]
    ntypes = sim.var('ntypes')
    return sim.array('damping_n')[type_a * ntypes + type_b]

def getDampingT(sim, params):
    type_a = params[0]
    type_b = params[1]
    ntypes = sim.var('ntypes')
    return sim.array('damping_t')[type_a * ntypes + type_b]

def getFriction(sim, params):
    type_a = params[0]
    type_b = params[1]
    ntypes = sim.var('ntypes')
    return sim.array('friction')[type_a * ntypes + type_b]

def getNormalizedOrZero(sim, params):
    vec = params[0]
    sqr_length = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
    double_prec = True
    epsilon = 1e-8 if double_prec else 1e-4
    return Select(
        sqr_length < epsilon * epsilon,
        vec, vec * (1.0 / Sqrt(sqr_length)))

def length(sim, params):
    vec = params[0]
    return Sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

def dot(sim, params):
    vec1 = params[0]
    vec2 = params[1]
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]

def map_kernel_to_simulation(sim, node):
    contactPoint = sim.add_var('contactPoint', Type_Vector)
    contactNormal = sim.add_var('contactNormal', Type_Vector)
    penetrationDepth = sim.add_var('penetrationDepth', Type_Float)

    for i, j in sim.particle_pairs():
        return map_method_tree(sim, node, {
            'element_mappings': {
                'p_idx1': i,
                'p_idx2': j,
                'ac': None,
                'contactPoint': contactPoint,
                'contactNormal': contactNormal,
                'penetrationDepth': penetrationDepth
            },
            'function_mappings': {
                'getVelocityAtWFPoint': getVelocityAtWFPoint,
                'getType': getType,
                'getNormalizedOrZero': getNormalizedOrZero,
                'getStiffness': getStiffness,
                'getDampingN': getDampingN,
                'getDampingT': getDampingT,
                'getFriction': getFriction,
                'length': length,
                'math::dot': dot
            }
        })

def map_method_tree(sim, node, assignments={}, mappings={}):
    if node is not None:
        if node.kind == kind.FUNCTION_TEMPLATE:
            for child in node.get_children():
                if child.kind == kind.PARAM_DECL:
                    type_ref = child.get_children()[0]
                    assert type_ref.kind == kind.TYPE_REF, \
                        "Expected type reference!"

                if child.kind == kind.COMPOUND_STMT:
                    return map_method_tree(sim, child, assignments, mappings)

        if node.kind == kind.COMPOUND_STMT:
            block = Block([])
            for child in node.get_children():
                stmt = map_method_tree(sim, child, assignments, mappings)
                if stmt is not None:
                    block.add(stmt)

            return block

        if node.kind == kind.DECL_STMT:
            child = node.get_children()[0]
            if child.kind == kind.VAR_DECL:
                #decl_type = child.get_children()[0]
                decl_expr = child.get_children()[1]
                assignments[child.spelling] = \
                    map_expression(sim, decl_expr, assignments, mappings)

            return None

        if node.kind == kind.IF_STMT:
            cond = map_expression(
                sim, node.get_children()[0], assignments, mappings)
            block_if_stmt = map_method_tree(
                sim, node.get_children()[0], assignments, mappings)
            return Branch(sim, cond, True, block_if_stmt, None)

        if node.kind == kind.RETURN_STMT:
            return None

        if node.kind == kind.CALL_EXPR:
            return map_call(sim, node, assignments, mappings)

    return None

def map_call(sim, node, assignments, mappings):
    func_name = None
    params = []

    for child in node.get_children():
        if child.kind == kind.DECL_REF_EXPR:
            grandchild = child.get_children()[0]
            namespace = map_namespace(grandchild)
            func_name = grandchild.spelling if namespace is None \
                        else f"{namespace}::{grandchild.spelling}"

        if child.kind == kind.MEMBER_REF_EXPR:
            params.append(map_expression(
                    sim, child.get_children()[0], assignments, mappings))

        else:
            params.append(map_expression(sim, child, assignments, mappings))

    assert func_name in mappings['function_mappings'], \
        f"No mapping for function: {func_name}"
    return mappings['function_mappings'][func_name](sim, params)

def map_namespace(node):
    namespace = None
    children = node.get_children()
    while children is not None and len(children) > 0:
        child = children[0]
        if child.kind == kind.NAMESPACE_REF:
            namespace = child.spelling if namespace is None \
                        else f"{child.spelling}::{namespace}"

        children = child.get_children()

    return namespace

def map_expression(sim, node, assignments, mappings):
    if node.kind == kind.UNEXPOSED_EXPR:
        return map_expression(
            sim, node.get_children()[0], assignments, mappings)

    if node.kind == kind.DECL_REF_EXPR:
        if node.spelling in assignments:
            return assignments[node.spelling]

        if node.spelling in mappings:
            return mappings[node.spelling]

    if node.kind == kind.CALL_EXPR:
        return map_call(sim, node, assignments, mappings)

    return None

def parse_walberla_file(filename):
    walberla_path = "/home/rzlin/az16ahoq/repositories/walberla"
    walberla_src = f"{walberla_path}/src"
    walberla_build_src = f"{walberla_path}/build/src"
    clang_include_path = "/software/anydsl/llvm_build/lib/clang/7.0.1/include"
    mpi_include_path = "/software/openmpi/4.0.0-llvm/include"
    
    index = clang.cindex.Index.create()
    tu = index.parse(
        f"{walberla_src}/{filename}",
        args = [
            "-Wall",
            f"-I{walberla_src}",
            f"-I{walberla_build_src}",
            f"-I{clang_include_path}",
            f"-I{mpi_include_path}"
        ]
    )

    diagnostics = tu.diagnostics
    for i in range(0, len(diagnostics)):
        print(diagnostics[i])

    print(f"Translation unit: {tu.spelling}")
    return tu
