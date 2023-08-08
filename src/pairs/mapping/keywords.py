from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.const_vector import ZeroVector
from pairs.ir.lit import Lit
from pairs.ir.loops import Continue
from pairs.ir.math import Sqrt
from pairs.ir.select import Select
from pairs.ir.types import Types


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

    def keyword_select(self, args):
        assert len(args) == 3, "select() keyword requires three parameters!"
        return Select(self.sim, args[0], args[1], args[2])

    def keyword_skip_when(self, args):
        assert len(args) == 1, "skip_when() keyword requires one parameter!"
        return Filter(self.sim, args[0], Block(self.sim, [Continue(self.sim)]))

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

    def keyword_normalized(self, args):
        assert len(args) == 1, "normalized() keyword requires one parameter!"
        vector = args[0]
        assert vector.type() == Types.Vector, "normalized(): Argument must be a vector!"
        length = self.keyword_length([vector])
        inv_length = Lit(self.sim, 1.0) / length
        return Select(self.sim, length > Lit(self.sim, 0.0), vector * inv_length, ZeroVector(self.sim))

    def keyword_squared_length(self, args):
        assert len(args) == 1, "length() keyword requires one parameter!"
        vector = args[0]
        assert vector.type() == Types.Vector, "length(): Argument must be a vector!"
        return sum([vector[d] * vector[d] for d in range(self.sim.ndims())])

    def keyword_zero_vector(self, args):
        assert len(args) == 0, "zero_vector() keyword requires no parameter!"
        return ZeroVector(self.sim)
