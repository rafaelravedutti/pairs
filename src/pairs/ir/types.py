class Types:
    Invalid = -1
    Int32 = 0
    Int64 = 1
    UInt64 = 2
    Float = 3
    Double = 4
    Boolean = 5
    String = 6
    Vector = 7
    Array = 8

    def ctype2keyword(t):
        return (
            'double' if t == Types.Double or t == Types.Vector
            else 'float' if t == Types.Float
            else 'int' if t == Types.Int32
            else 'long long int' if t == Types.Int64
            else 'unsigned long long int' if t == Types.UInt64
            else 'bool'
        )

    def is_integer(t):
        return t == Types.Int32 or t == Types.Int64 or t == Types.UInt64

    def is_real(t):
        return t == Types.Float or t == Types.Double
