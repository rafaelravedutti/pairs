class Types:
    Invalid = -1
    Int32 = 0
    Int64 = 1
    UInt64 = 2
    Real = 3
    Float = 4
    Double = 5
    Boolean = 6
    String = 7
    Vector = 8
    Array = 9
    Matrix = 10
    Quaternion = 11

    def c_keyword(sim, t):
        real_kw = 'double' if sim.use_double_precision() else 'float'
        return (
            real_kw if t in (Types.Real, Types.Vector, Types.Matrix, Types.Quaternion)
            else 'float' if t == Types.Float
            else 'double' if t == Types.Double
            else 'int' if t == Types.Int32
            else 'long long int' if t == Types.Int64
            else 'unsigned long long int' if t == Types.UInt64
            else 'bool' if t == Types.Boolean
            else '<invalid type>'
        )

    def c_property_keyword(t):
        return "Prop_Integer"      if t == Types.Int32 else \
               "Prop_Real"         if t == Types.Real else \
               "Prop_Vector"       if t == Types.Vector else \
               "Prop_Matrix"       if t == Types.Matrix else \
               "Prop_Quaternion"   if t == Types.Quaternion else \
               "Prop_Invalid"

    def is_integer(t):
        return t in (Types.Int32, Types.Int64, Types.UInt64)

    def is_real(t):
        return t in (Types.Float, Types.Double, Types.Real)

    def is_scalar(t):
        return t not in (Types.Vector, Types.Matrix, Types.Quaternion)

    def number_of_elements(sim, t):
        return sim.ndims() if t == Types.Vector else \
               sim.ndims() * sim.ndims() if t == Types.Matrix else \
               sim.ndims() + 1 if t == Types.Quaternion else \
               1
