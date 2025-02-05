from pairs.ir.types import Types

class PairsAcessor:
    def __init__(self, cgen):
        self.sim = cgen.sim
        self.target = cgen.target
        self.print = cgen.print
        self.debug = cgen.debug
        self.host_device_attr = ""
        self.host_attr = ""
        
    def generate(self):
        self.print("")

        if self.target.is_gpu():
            self.host_device_attr = "__host__ __device__ "
            self.host_attr = "__host__ "
        self.print("#include \"runtime/math/Vector3.hpp\"")
        # self.print("#include \"runtime/math/Quaternion.hpp\"")
        # self.print("#include \"runtime/math/Matrix3.hpp\"")
        self.print("")

        self.print("class PairsAccessor {")
        self.print("private:")
        self.print.add_indent(4)
        self.member_variables()
        self.print.add_indent(-4)
        self.print("public:")
        self.print.add_indent(4)

        if self.target.is_gpu():
            self.update()

        self.constructor()

        for p in self.sim.properties:
            if (p.type()==Types.Vector) or (Types.is_scalar(p.type())):
                self.get_property(p)    
                self.set_property(p)
                if self.target.is_gpu():
                        self.sync_property(p)

        self.utility_funcs()
            
        self.print.add_indent(-4)
        self.print("};")
        self.print("")

    def member_variables(self):
        self.print("PairsSimulation *ps;")

        if self.target.is_gpu():
            self.print("int *nlocal_d;")
            self.print("int *nghost_d;")
            self.print("")

            self.print("//Properties")
            for p in self.sim.properties:
                pname = p.name()
                tkw = Types.c_keyword(self.sim, p.type())
                self.print(f"{tkw} *{pname}_d;")

            self.print("")
            self.print("//Property flags")
            for p in self.sim.properties:
                pname = p.name()
                tkw = Types.c_keyword(self.sim, Types.Boolean)
                self.print(f"{tkw} *{pname}_device_flag_d;")
                self.print(f"{tkw} {pname}_device_flag_h = false;")
                self.print(f"{tkw} {pname}_host_flag = false;")

        self.print("")

    def update(self):
        self.print(f"{self.host_attr}void update(){{")
        self.print.add_indent(4)
        self.print(f"cudaMemcpy(nlocal_d, &(ps->pobj->nlocal), sizeof(int), cudaMemcpyHostToDevice);")
        self.print(f"cudaMemcpy(nghost_d, &(ps->pobj->nghost), sizeof(int), cudaMemcpyHostToDevice);")

        for p in self.sim.properties:
            pname = p.name()
            self.print(f"{pname}_d = ps->pobj->{pname}_d;")

        self.print.add_indent(-4)
        self.print("}")
        self.print("")


    def constructor(self):
        if self.target.is_gpu():
            self.print(f"{self.host_attr}PairsAccessor(PairsSimulation *ps_): ps(ps_){{")
            self.print.add_indent(4)

            self.print(f"cudaMalloc(&nlocal_d, sizeof(int));")
            self.print(f"cudaMalloc(&nghost_d, sizeof(int));")
            self.print("this->update();")

            for p in self.sim.properties:
                pname = p.name()
                tkw = Types.c_keyword(self.sim, Types.Boolean)
                self.print(f"cudaMalloc(&{pname}_device_flag_d, sizeof({tkw}));")
                self.print(f"cudaMemcpy({pname}_device_flag_d, &{pname}_device_flag_h, sizeof({tkw}), cudaMemcpyHostToDevice);")
        
            self.print.add_indent(-4)
            self.print("}")
        else:
            self.print("PairsAccessor(PairsSimulation *ps_): ps(ps_){}")

        self.print("")
    
    def ifdef_else(self, ifdef, func1, args1, func2, args2):
        self.print.add_indent(4)
        self.print(f"#ifdef {ifdef}")
        func1(*args1)
        self.print("#else")
        func2(*args2)
        self.print("#endif")
        self.print.add_indent(-4)

    def getter_body(self, prop, device=False):
        self.print.add_indent(4)
        pname = prop.name()
        tkw = Types.c_accessor_keyword(self.sim, prop.type())
        
        if self.target.is_gpu() and device:
            v = f"{pname}_d"
        else:
            v = f"ps->pobj->{pname}"

        if Types.is_scalar(prop.type()):
            self.print(f"return {v}[i];")
        else:
            nelems = Types.number_of_elements(self.sim, prop.type())
            return_values = [f"{v}[i*{nelems} + {n}]" for n in range(nelems)] 
            self.print(f"return {tkw}(" + ", ".join(rv for rv in return_values) + ");")
        self.print.add_indent(-4)


    def get_property(self, prop):
        pname = prop.name()
        tkw = Types.c_accessor_keyword(self.sim, prop.type())
        splitname = pname.split('_')
        funcname = ''.join(word.capitalize() for word in splitname)

        self.print(f"{self.host_device_attr}{tkw} get{funcname}(const size_t i) const{{")

        if self.target.is_gpu():
            self.ifdef_else("__CUDA_ARCH__", self.getter_body, [prop, True], self.getter_body, [prop, False])
        else:
            self.getter_body(prop, False)

        self.print("}")
        self.print("")


    def setter_body(self, prop, device=False):
        self.print.add_indent(4)
        pname = prop.name()
        tkw = Types.c_accessor_keyword(self.sim, prop.type())

        if self.target.is_gpu() and device:
            v = f"{pname}_d"
        else:
            v = f"ps->pobj->{pname}"

        if Types.is_scalar(prop.type()):
            self.print(f"{v}[i] = value;")
        else:
            nelems = Types.number_of_elements(self.sim, prop.type())
            set_values = [f"{v}[i*{nelems} + {n}] = value[{n}];" for n in range(nelems)] 
            for sv in set_values:
                self.print(sv)

        if self.target.is_gpu():
            flag = f"*{pname}_device_flag_d" if device else f"{pname}_host_flag"
            self.print(f"{flag} = true;")

        self.print.add_indent(-4)


    def set_property(self, prop):
        pname = prop.name()
        tkw = Types.c_accessor_keyword(self.sim, prop.type())
        splitname = pname.split('_')
        funcname = ''.join(word.capitalize() for word in splitname)

        self.print(f"{self.host_device_attr}void set{funcname}(const size_t i, const {tkw} &value){{")

        if self.target.is_gpu():
            self.ifdef_else("__CUDA_ARCH__", self.setter_body, [prop, True], self.setter_body, [prop, False])
        else:
            self.setter_body(prop, False)

        self.print("}")
        self.print("")

    def sync_property(self, prop):
        pname = prop.name()
        pid = prop.id()
        splitname = pname.split('_')
        funcname = ''.join(word.capitalize() for word in splitname)

        self.print(f"{self.host_attr}void sync{funcname}(){{")
        self.print.add_indent(4)
        self.print(f"{pname}_d = ps->pobj->{pname}_d;")
        self.print(f"cudaMemcpy(&{pname}_device_flag_h, {pname}_device_flag_d, sizeof(bool), cudaMemcpyDeviceToHost);")
        self.print("")
        

        #####################################################################################################################
        #####################################################################################################################
        # self.print(f"if (({pname}_host_flag && {pname}_device_flag_h) || ")
        # self.print.add_indent(4)
        # self.print(f"({pname}_host_flag && !ps->pairs_runtime->getPropFlags()->isHostFlagSet({pid})) ||")
        # self.print(f"({pname}_device_flag_h && !ps->pairs_runtime->getPropFlags()->isDeviceFlagSet({pid}))){{")
        # self.print(f"PAIRS_ERROR(\"OUT OF SYNC! Both host and device versions of {pname} are in a modified state.\\n\");")
        # self.print("exit(-1);")
        # self.print.add_indent(-4)
        # self.print("}")
        # self.print("")


        self.print(f"if ({pname}_host_flag && {pname}_device_flag_h){{")
        self.print.add_indent(4)
        self.print(f"PAIRS_ERROR(\"OUT OF SYNC 1! Both host and device versions of {pname} are in a modified state.\\n\");")
        self.print("exit(-1);")
        self.print.add_indent(-4)
        self.print("}")
        self.print("")

        self.print(f"if ({pname}_host_flag && !ps->pairs_runtime->getPropFlags()->isHostFlagSet({pid})){{")
        self.print.add_indent(4)
        self.print(f"PAIRS_ERROR(\"OUT OF SYNC 2! Both host and device versions of {pname} are in a modified state.\\n\");")
        self.print("exit(-1);")
        self.print.add_indent(-4)
        self.print("}")
        self.print("")

        self.print(f"if ({pname}_device_flag_h && !ps->pairs_runtime->getPropFlags()->isDeviceFlagSet({pid})){{")
        self.print.add_indent(4)
        self.print(f"PAIRS_ERROR(\"OUT OF SYNC 3! Both host and device versions of {pname} are in a modified state.\\n\");")
        self.print("exit(-1);")
        self.print.add_indent(-4)
        self.print("}")
        self.print("")

        #####################################################################################################################
        #####################################################################################################################


        self.print(f"if ({pname}_host_flag){{")
        self.print.add_indent(4)
        self.print(f"ps->pairs_runtime->getPropFlags()->setHostFlag({pid});")
        self.print(f"ps->pairs_runtime->getPropFlags()->clearDeviceFlag({pid});")
        self.print.add_indent(-4)
        self.print("}")
        
        self.print(f"else if ({pname}_device_flag_h){{")
        self.print.add_indent(4)
        self.print(f"ps->pairs_runtime->getPropFlags()->setDeviceFlag({pid});")
        self.print(f"ps->pairs_runtime->getPropFlags()->clearHostFlag({pid});")
        self.print.add_indent(-4)
        self.print("}")
        self.print("")

        nelems = Types.number_of_elements(self.sim, prop.type())
        tkw = Types.c_keyword(self.sim, prop.type())

        self.print(f"if (ps->pairs_runtime->getPropFlags()->isHostFlagSet({pid})) {{")
        self.print.add_indent(4)

        self.print(f"ps->pairs_runtime->copyPropertyToDevice({pid}, ReadOnly, (((ps->pobj->nlocal + ps->pobj->nghost) * {nelems}) * sizeof({tkw})));")
        self.print.add_indent(-4)
        self.print("}")

        self.print(f"else if (ps->pairs_runtime->getPropFlags()->isDeviceFlagSet({pid})) {{")
        self.print.add_indent(4)
        self.print(f"ps->pairs_runtime->copyPropertyToHost({pid}, ReadOnly, (((ps->pobj->nlocal + ps->pobj->nghost) * {nelems}) * sizeof({tkw})));")
        self.print.add_indent(-4)
        self.print("}")
        self.print("")

        self.print(f"{pname}_host_flag = false;")
        self.print(f"{pname}_device_flag_h = false;")
        self.print(f"cudaMemcpy({pname}_device_flag_d, &{pname}_device_flag_h, sizeof(bool), cudaMemcpyHostToDevice);")

        self.print.add_indent(-4)
        self.print("}")
        self.print("")

    def utility_funcs(self):
        if self.target.is_gpu():
            self.print(f"{self.host_device_attr}int size() const {{")
            self.print("    #ifdef __CUDA_ARCH__")
            self.print("        return *nlocal_d + *nghost_d;")
            self.print("    #else")
            self.print("        return ps->pobj->nlocal + ps->pobj->nghost;")
            self.print("    #endif")
            self.print("}")
            self.print("")
        else:
            self.print("int size() const {return ps->pobj->nlocal + ps->pobj->nghost;}")

        if self.target.is_gpu():
            self.print(f"{self.host_device_attr}int nlocal() const {{")
            self.print("    #ifdef __CUDA_ARCH__")
            self.print("        return *nlocal_d;")
            self.print("    #else")
            self.print("        return ps->pobj->nlocal;")
            self.print("    #endif")
            self.print("}")
            self.print("")
        else:
            self.print("int nlocal() const {return ps->pobj->nlocal;}")

        if self.target.is_gpu():
            self.print(f"{self.host_device_attr}int nghost() const {{")
            self.print("    #ifdef __CUDA_ARCH__")
            self.print("        return *nghost_d;")
            self.print("    #else")
            self.print("        return ps->pobj->nghost;")
            self.print("    #endif")
            self.print("}")
            self.print("")
        else:
            self.print("int nghost() const {return ps->pobj->nghost;}")


        self.print(f"{self.host_device_attr}int getInvalidIdx(){{return -1;}}")
        self.print("")

        self.print(f"{self.host_device_attr}pairs::id_t getInvalidUid(){{return 0;}}")
        self.print("")

        self.print(f"{self.host_device_attr}int uidToIdx(pairs::id_t uid){{")
        self.print("    int idx = getInvalidIdx();")
        self.print("    for(int i=0; i<size(); ++i){")
        self.print("        if (getUid(i) == uid){")
        self.print("            idx = i;")
        self.print("            break;")
        self.print("        }")
        self.print("    }")
        self.print("    return idx;")
        self.print("}")
        self.print("")

        self.print(f"{self.host_device_attr}int uidToIdxLocal(pairs::id_t uid){{")
        self.print("    int idx = getInvalidIdx();")
        self.print("    for(int i=0; i<nlocal(); ++i){")
        self.print("        if (getUid(i) == uid){")
        self.print("            idx = i;")
        self.print("            break;")
        self.print("        }")
        self.print("    }")
        self.print("    return idx;")
        self.print("}")
        self.print("")

        self.print(f"{self.host_device_attr}int uidToIdxGhost(pairs::id_t uid){{")
        self.print("    int idx = getInvalidIdx();")
        self.print("    for(int i=nlocal(); i<size(); ++i){")
        self.print("        if (getUid(i) == uid){")
        self.print("            idx = i;")
        self.print("            break;")
        self.print("        }")
        self.print("    }")
        self.print("    return idx;")
        self.print("}")
        self.print("")
