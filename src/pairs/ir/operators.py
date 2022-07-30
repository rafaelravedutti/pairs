class Op:
    def __init__(self, symbol, conditional=False, boolean=False, unary=False):
        self._symbol = symbol
        self._conditional = conditional
        self._boolean = boolean
        self._unary = unary

    def symbol(self):
        return self._symbol

    def is_conditional(self):
        return self._conditional

    def is_boolean(self):
        return self._boolean

    def is_unary(self):
        return self._unary


class Operators:
    Invalid =   Op("<invalid>")
    Add     =   Op('+')
    Sub     =   Op('-')
    Mul     =   Op('*')
    Div     =   Op('/')
    Mod     =   Op('%')
    BinAnd  =   Op('&')
    BinOr   =   Op('|')
    BinXor  =   Op('^')
    BinNeg  =   Op('~',  unary=True)
    Eq      =   Op('==', conditional=True)
    Neq     =   Op('!=', conditional=True)
    Lt      =   Op('<',  conditional=True)
    Leq     =   Op('<=', conditional=True)
    Gt      =   Op('>',  conditional=True)
    Geq     =   Op('>=', conditional=True)
    And     =   Op('&&', boolean=True)
    Or      =   Op('||', boolean=True)
