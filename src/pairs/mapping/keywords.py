from pairs.ir.apply import Apply
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.lit import Lit
from pairs.ir.loops import Continue
from pairs.ir.math import Abs, Cos, Sin, Sqrt
from pairs.ir.matrices import Matrix
from pairs.ir.properties import Property
from pairs.ir.quaternions import Quaternion
from pairs.ir.scalars import ScalarOp
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.ir.vectors import Vector, ZeroVector
from pairs.sim.shapes import Shapes


class Keywords:
    def __init__(self, sim):
        self.sim = sim

    def get_method(self, method_name):
        method = getattr(self, method_name, None)
        return method if callable(method) else None

    def __call__(self, keyword, args):
        method = self.get_method(f"keyword_{keyword}")
        assert method is not None, "Invalid keyword: {keyword}"
        return method(args)

    def exists(self, keyword):
        method = self.get_method(f"keyword_{keyword}")
        return method is not None

    def keyword_is_point_mass(self, args):
        assert len(args) == 1, "is_point_mass() keyword requires one parameter."
        particle_id = args[0]
        assert particle_id.type() == Types.Int32, "Particle ID must be an integer."
        return ScalarOp.cmp(self.sim.particle_shape[particle_id], Shapes.PointMass)

    def keyword_is_sphere(self, args):
        assert len(args) == 1, "is_sphere() keyword requires one parameter."
        particle_id = args[0]
        assert particle_id.type() == Types.Int32, "Particle ID must be an integer."
        return ScalarOp.cmp(self.sim.particle_shape[particle_id], Shapes.Sphere)

    def keyword_is_halfspace(self, args):
        assert len(args) == 1, "is_sphere() keyword requires one parameter."
        particle_id = args[0]
        assert particle_id.type() == Types.Int32, "Particle ID must be an integer."
        return ScalarOp.cmp(self.sim.particle_shape[particle_id], Shapes.Halfspace)

    def keyword_select(self, args):
        assert len(args) == 3, "select() keyword requires three parameters!"
        return Select(self.sim, args[0], args[1], args[2])

    def keyword_sqrt(self, args):
        assert len(args) == 1, "sqrt() keyword requires one parameter!"
        value = args[0]
        assert value.type() == Types.Real, "sqrt(): Value must be a real."
        return Sqrt(self.sim, value)

    def keyword_skip_when(self, args):
        assert len(args) == 1, "skip_when() keyword requires one parameter!"
        for _ in Filter(self.sim, args[0]):
            Continue(self.sim)()

    def keyword_min(self, args):
        e_min = args[0]
        for a in args[1:]:
            e_min = Select(self.sim, a < e_min, a, e_min)

        return e_min

    def keyword_max(self, args):
        e_max = args[0]
        for a in args[1:]:
            e_max = Select(self.sim, a > e_max, a, e_max)

        return e_max

    def keyword_length(self, args):
        assert len(args) == 1, "length() keyword requires one parameter!"
        vector = args[0]
        assert vector.type() == Types.Vector, "length(): Argument must be a vector!"
        return Sqrt(self.sim, sum([vector[d] * vector[d] for d in range(self.sim.ndims())]))

    def keyword_dot(self, args):
        assert len(args) == 2, "dot() keyword requires two parameters!"
        vector1 = args[0]
        vector2 = args[1]
        assert vector1.type() == Types.Vector, "dot(): First argument must be a vector!"
        assert vector2.type() == Types.Vector, "dot(): Second argument must be a vector!"
        return sum([vector1[d] * vector2[d] for d in range(self.sim.ndims())])

    def keyword_cross(self, args):
        assert len(args) == 2, "cross() keyword requires two parameters!"
        vector1 = args[0]
        vector2 = args[1]
        assert vector1.type() == Types.Vector, "cross(): First argument must be a vector!"
        assert vector2.type() == Types.Vector, "cross(): Second argument must be a vector!"
        return Vector(self.sim, [ vector1[1] * vector2[2] - vector1[2] * vector2[1],
                                  vector1[2] * vector2[0] - vector1[0] * vector2[2],
                                  vector1[0] * vector2[1] - vector1[1] * vector2[0] ])

    def keyword_normalized(self, args):
        assert len(args) == 1, "normalized() keyword requires one parameter!"
        vector = args[0]
        assert vector.type() == Types.Vector, "normalized(): Argument must be a vector!"
        length = self.keyword_length([vector])
        inv_length = Lit(self.sim, 1.0) / length
        return Select(self.sim, length > Lit(self.sim, 0.0), vector * inv_length, ZeroVector(self.sim))
        #return vector * inv_length

    def keyword_squared_length(self, args):
        assert len(args) == 1, "length() keyword requires one parameter."
        vector = args[0]
        assert vector.type() == Types.Vector, "length(): Argument must be a vector."
        return sum([vector[d] * vector[d] for d in range(self.sim.ndims())])

    def keyword_zero_vector(self, args):
        assert len(args) == 0, "zero_vector() keyword requires no parameter."
        return ZeroVector(self.sim)

    def keyword_transposed(self, args):
        assert len(args) == 1, "transposed() keyword requires one parameter."
        matrix = args[0]
        assert matrix.type() == Types.Matrix, "tranposed(): Argument must be a matrix."
        return Matrix(self.sim, [ matrix[0], matrix[3], matrix[6],
                                  matrix[1], matrix[4], matrix[7],
                                  matrix[2], matrix[5], matrix[8] ])

    def keyword_inversed(self, args):
        assert len(args) == 1, "inversed() keyword requires one parameter."
        matrix = args[0]
        assert matrix.type() == Types.Matrix, "inversed(): Argument must be a matrix."
        det = matrix[0] * ((matrix[4] * matrix[8]) - (matrix[7] * matrix[5])) + \
              matrix[1] * ((matrix[5] * matrix[6]) - (matrix[8] * matrix[3])) + \
              matrix[2] * ((matrix[3] * matrix[7]) - (matrix[4] * matrix[6]))

        inv_det = 1.0 / det
        return Matrix(self.sim, [ inv_det * ((matrix[4] * matrix[8]) - (matrix[5] * matrix[7])),
                                  inv_det * ((matrix[7] * matrix[2]) - (matrix[8] * matrix[1])),
                                  inv_det * ((matrix[1] * matrix[5]) - (matrix[2] * matrix[4])),
                                  inv_det * ((matrix[5] * matrix[6]) - (matrix[3] * matrix[8])),
                                  inv_det * ((matrix[8] * matrix[0]) - (matrix[6] * matrix[2])),
                                  inv_det * ((matrix[2] * matrix[3]) - (matrix[0] * matrix[5])),
                                  inv_det * ((matrix[3] * matrix[7]) - (matrix[4] * matrix[6])),
                                  inv_det * ((matrix[6] * matrix[1]) - (matrix[7] * matrix[0])),
                                  inv_det * ((matrix[0] * matrix[4]) - (matrix[1] * matrix[3])) ])

    def keyword_diagonal_matrix(self, args):
        assert len(args) == 1, "diagonal_matrix() keyword requires one parameter!"
        value = args[0]
        nelems = Types.number_of_elements(self.sim, Types.Matrix)
        return Matrix(self.sim, [value if i % (self.sim.ndims() + 1) == 0 else 0.0 \
                                 for i in range(nelems)])

    def keyword_matrix_multiplication(self, args):
        assert len(args) == 2, "matrix_multiplication() keyword requires two parameters!"
        lhs = args[0]
        rhs = args[1]
        nelems = Types.number_of_elements(self.sim, Types.Matrix)
        assert Types.Matrix in (lhs.type(), rhs.type()), \
            "matrix_multiplication(): At least one matrix is needed!"

        # Matrix * Matrix
        if lhs.type() == rhs.type():
            return Matrix(self.sim, [ lhs[0] * rhs[0] + lhs[1] * rhs[3] + lhs[2] * rhs[6],
                                      lhs[0] * rhs[1] + lhs[1] * rhs[4] + lhs[2] * rhs[7],
                                      lhs[0] * rhs[2] + lhs[1] * rhs[5] + lhs[2] * rhs[8],
                                      lhs[3] * rhs[0] + lhs[4] * rhs[3] + lhs[5] * rhs[6],
                                      lhs[3] * rhs[1] + lhs[4] * rhs[4] + lhs[5] * rhs[7],
                                      lhs[3] * rhs[2] + lhs[4] * rhs[5] + lhs[5] * rhs[8],
                                      lhs[6] * rhs[0] + lhs[7] * rhs[3] + lhs[8] * rhs[6],
                                      lhs[6] * rhs[1] + lhs[7] * rhs[4] + lhs[8] * rhs[7],
                                      lhs[6] * rhs[2] + lhs[7] * rhs[5] + lhs[8] * rhs[8] ])

        if Types.Vector in (lhs.type(), rhs.type()):
            # Matrix * Vector
            if lhs.type() == Types.Matrix:
                return Vector(self.sim, [ lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2],
                                          lhs[3] * rhs[0] + lhs[4] * rhs[1] + lhs[5] * rhs[2], 
                                          lhs[6] * rhs[0] + lhs[7] * rhs[1] + lhs[8] * rhs[2] ])

            # Vector * Matrix
            else:
                return Vector(self.sim, [ lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2],
                                          lhs[0] * rhs[3] + lhs[1] * rhs[4] + lhs[2] * rhs[5], 
                                          lhs[0] * rhs[6] + lhs[1] * rhs[7] + lhs[2] * rhs[8] ])

        # Scalar * Matrix
        if rhs.type() == Types.Matrix:
            return Matrix(self.sim, [rhs[i] * lhs for i in range(nelems)])

        # Matrix * Scalar
        return Matrix(self.sim, [lhs[i] * rhs for i in range(nelems)])

    def keyword_default_quaternion(self, args):
        assert len(args) == 0, "default_quaternion() requires no parameters!"
        return Quaternion(self.sim, [1.0, 0.0, 0.0, 0.0])

    def keyword_quaternion(self, args):
        assert len(args) == 2, "quaternion() keyword requires two parameters!"
        axis = args[0]
        angle = args[1]
        epsilon = 1e-6
        assert axis.type() == Types.Vector, "quaternion(): First argument must be a vector."
        assert Types.is_real(angle.type()), "quaternion(): Second argument must be a real value."

        axis_length = self.keyword_length([axis])
        zero_cond = Abs(self.sim, axis_length) < epsilon or Abs(self.sim, angle) < epsilon
        sina = Sin(self.sim, angle * 0.5)
        cosa = Cos(self.sim, angle * 0.5)
        axisN = axis * (1.0 / axis_length)
        return Select(self.sim, zero_cond,
                      Quaternion(self.sim, [1.0, 0.0, 0.0, 0.0]),
                      Quaternion(self.sim, [cosa, sina * axisN[0], sina * axisN[1], sina * axisN[2]]))

    def keyword_quaternion_multiplication(self, args):
        assert len(args) == 2, "quaternion_multiplication() keyword requires two parameters!"
        lhs = args[0]
        rhs = args[1]
        assert lhs.type() == Types.Quaternion, \
            "quaternion_multiplication(): Left-hand side operator is not a quaternion!"
        assert rhs.type() == Types.Quaternion, \
            "quaternion_multiplication(): Right-hand side operator is not a quaternion!"

        r = lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3]
        i = lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] - lhs[3] * rhs[2]
        j = lhs[0] * rhs[2] + lhs[2] * rhs[0] + lhs[3] * rhs[1] - lhs[1] * rhs[3]
        k = lhs[0] * rhs[3] + lhs[3] * rhs[0] + lhs[1] * rhs[2] - lhs[2] * rhs[1]

        len2 = r * r + i * i + j * j + k * k
        ilen = Select(self.sim, Abs(self.sim, len2 - 1.0) < 1E-8, 1.0, 1.0 / Sqrt(self.sim, len2))
        return Quaternion(self.sim, [r * ilen, i * ilen, j * ilen, k * ilen])

    def keyword_quaternion_to_rotation_matrix(self, args):
        assert len(args) == 1, "quaternion_to_rotation_matrix() keyword requires one parameter!"
        quat = args[0]
        assert quat.type() == Types.Quaternion, \
            "quaternion_to_rotation_matrix(): Given argument is not a quaternion!"

        return Matrix(self.sim, [ 1.0 - 2.0 * quat[2] * quat[2] - 2.0 * quat[3] * quat[3],
                                  2.0 * (quat[1] * quat[2] - quat[0] * quat[3]),
                                  2.0 * (quat[1] * quat[3] + quat[0] * quat[2]),
                                  2.0 * (quat[1] * quat[2] + quat[0] * quat[3]),
                                  1.0 - 2.0 * quat[1] * quat[1] - 2.0 * quat[3] * quat[3],
                                  2.0 * (quat[2] * quat[3] - quat[0] * quat[1]),
                                  2.0 * (quat[1] * quat[3] - quat[0] * quat[2]),
                                  2.0 * (quat[2] * quat[3] + quat[0] * quat[1]),
                                  1.0 - 2.0 * quat[1] * quat[1] - 2.0 * quat[2] * quat[2] ])

    def keyword_apply(self, args):
        assert len(args) == 3, "apply() keyword requires two parameters."
        prop = args[0]
        expr = args[1]
        assert isinstance(prop, Property), "apply(): First argument must be a property."
        assert prop.type() == expr.type(), "apply(): Property and expression must be of same type."
        Apply(self.sim, prop, expr, args[2])
