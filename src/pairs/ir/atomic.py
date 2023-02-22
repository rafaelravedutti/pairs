from pairs.ir.bin_op import ASTTerm, BinOp
from pairs.ir.lit import Lit


class AtomicAdd(ASTTerm):
    last_atomic_add = 0

    def new_id():
        AtomicAdd.last_atomic_add += 1
        return AtomicAdd.last_atomic_add - 1

    def __init__(self, sim, elem, value):
        super().__init__(sim)
        self.atomic_add_id = AtomicAdd.new_id()
        self.elem = BinOp.inline(elem)
        self.value = Lit.cvt(sim, value)
        self.resize = None
        self.capacity = None
        self.device_flag = False

    def __str__(self):
        return f"AtomicAdd<{self.elem, self.val}>"

    def add_resize_check(self, resize, capacity):
        self.resize = BinOp.inline(resize)
        self.capacity = capacity

    def check_for_resize(self):
        return self.resize is not None

    def id(self):
        return self.atomic_add_id

    def type(self):
        return self.elem.type()

    def children(self):
        return  [self.elem, self.value] + \
                ([self.resize, self.capacity] if self.resize is not None else [])
