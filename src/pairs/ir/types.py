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
    Matrix = 9
    Quaternion = 10

    def c_keyword(t):
        return (
            'double' if t in (Types.Double, Types.Vector, Types.Matrix, Types.Quaternion)
            else 'float' if t == Types.Float
            else 'int' if t == Types.Int32
            else 'long long int' if t == Types.Int64
            else 'unsigned long long int' if t == Types.UInt64
            else 'bool' if t == Types.Boolean
            else '<invalid type>'
        )

    def c_property_keyword(t):
        return "Prop_Integer"      if t == Types.Int32 else \
               "Prop_Float"        if t == Types.Double else \
               "Prop_Vector"       if t == Types.Vector else \
               "Prop_Matrix"       if t == Types.Matrix else \
               "Prop_Quaternion"   if t == Types.Quaternion else \
               "Prop_Invalid"

    def is_integer(t):
        return t in (Types.Int32, Types.Int64, Types.UInt64)

    def is_real(t):
        return t in (Types.Float, Types.Double)

    def is_scalar(t):
        return t not in (Types.Vector, Types.Matrix, Types.Quaternion)

    def number_of_elements(sim, t):
        return sim.ndims() if t == Types.Vector else \
               sim.ndims() * sim.ndims() if t == Types.Matrix else \
               sim.ndims() + 1 if t == Types.Quaternion else \
               1
