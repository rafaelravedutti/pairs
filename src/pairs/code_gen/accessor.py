from pairs.ir.types import Types
from pairs.ir.features import FeatureProperty
from pairs.ir.properties import Property

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
            self.print("namespace pairs::internal{")
            self.print.add_indent(4)
            self.DeviceProps_struct()
            self.HostProps_struct()
            self.print.add_indent(-4)
            self.print("}")
            self.print("")

        if self.target.is_gpu():
            self.host_device_attr = "__host__ __device__ "
            self.host_attr = "__host__ "
        self.print("#include \"math/Vector3.hpp\"")
        # self.print("#include \"math/Quaternion.hpp\"")
        # self.print("#include \"math/Matrix3.hpp\"")
        self.print("")

        self.print("class PairsAccessor {")
        self.print("private:")
        self.print.add_indent(4)
        self.member_variables()
        self.print.add_indent(-4)
        self.print("public:")
        self.print.add_indent(4)

        self.sync_ctx_enum()
        self.update()
        self.constructor()
        self.end()

        for p in self.sim.properties:
            if (p.type()==Types.Vector) or (Types.is_scalar(p.type())):
                self.get_property(p)    
                self.set_property(p)
                self.sync_property(p)

        for fp in self.sim.feature_properties:
            self.get_property(fp)
            self.set_property(fp)
            self.sync_feature_property(fp)

        self.utility_funcs()
            
        self.print.add_indent(-4)
        self.print("};")
        self.print("")

    def DeviceProps_struct(self):
        self.print("struct DeviceProps{")
        self.print.add_indent(4)

        self.print("int nlocal;")
        self.print("int nghost;")
        self.print("")

        self.print("//Property device pointers")
        for p in self.sim.properties:
            pname = p.name()
            tkw = Types.c_keyword(self.sim, p.type())
            self.print(f"{tkw} *{pname}_d;")

        self.print("")
        self.print("//Property device flag pointers")
        for p in self.sim.properties:
            pname = p.name()
            tkw = Types.c_keyword(self.sim, Types.Boolean)
            self.print(f"{tkw} *{pname}_device_flag_d;")

        self.print("")
        self.print("//Feature properties on device are global")

        self.print("")
        self.print("//Feature properties have no flags on device since they can't be modified on device")

        self.print.add_indent(-4)
        self.print("};")   
        self.print("")

    def HostProps_struct(self):
        self.print("// HostProps only contains property flags, since properties themselves can be directly accessed through ps->pobj")
        self.print("// TODO: Move properties out of PairsObjects into DeviceProps and HostProps, so that all 3 structs have mutually exclusive members")
        self.print("struct HostProps{")
        self.print.add_indent(4)

        self.print("")
        self.print("//Property host pointers are in PairsObjects")

        self.print("")
        self.print("//Property host flags")
        for p in self.sim.properties:
            pname = p.name()
            tkw = Types.c_keyword(self.sim, Types.Boolean)
            self.print(f"{tkw} {pname}_host_flag = false;")
        
        self.print("")
        self.print("//Property device flags")
        for p in self.sim.properties:
            pname = p.name()
            tkw = Types.c_keyword(self.sim, Types.Boolean)
            self.print(f"{tkw} {pname}_device_flag_h = false;")

        self.print("")
        self.print("//Feature property host pointers are in PairsObjects")

        self.print("")
        self.print("//Feature property host flags")
        for fp in self.sim.feature_properties:
            fpname = fp.name()
            tkw = Types.c_keyword(self.sim, Types.Boolean)
            self.print(f"{tkw} {fpname}_host_flag = false;")
        
        self.print("")
        self.print("//Feature properties have no device flags")

        self.print.add_indent(-4)
        self.print("};")
        self.print("")

    def member_variables(self):
        self.print("PairsSimulation *ps;")
        if self.target.is_gpu():
            self.print("pairs::internal::HostProps *hp;")
            self.print("pairs::internal::DeviceProps *dp_h;")
            self.print("pairs::internal::DeviceProps *dp_d;")

    def update(self):
        self.print(f"{self.host_attr}void update(){{")
        if self.target.is_gpu():
            self.print.add_indent(4)
            self.print(f"dp_h->nlocal = ps->pobj->nlocal;")
            self.print(f"dp_h->nghost = ps->pobj->nghost;")

            for p in self.sim.properties:
                pname = p.name()
                self.print(f"dp_h->{pname}_d = ps->pobj->{pname}_d;")

            self.print(f"cudaMemcpy(dp_d, dp_h, sizeof(pairs::internal::DeviceProps), cudaMemcpyHostToDevice);")
            self.print.add_indent(-4)
        self.print("}")
        self.print("")

    def constructor(self):
        if self.target.is_gpu():
            self.print(f"{self.host_attr}PairsAccessor(PairsSimulation *ps_): ps(ps_){{")
            self.print.add_indent(4)

            self.print(f"hp = new pairs::internal::HostProps;")
            self.print(f"dp_h = new pairs::internal::DeviceProps;")
            self.print(f"cudaMalloc(&dp_d, sizeof(pairs::internal::DeviceProps));")

            for p in self.sim.properties:
                pname = p.name()
                tkw = Types.c_keyword(self.sim, Types.Boolean)
                self.print(f"cudaMalloc(&(dp_h->{pname}_device_flag_d), sizeof({tkw}));")
        
            self.print("this->update();")
            self.print.add_indent(-4)
            self.print("}")

        else:
            self.print("PairsAccessor(PairsSimulation *ps_): ps(ps_){}")

        self.print("")

    def end(self):
        self.print(f"{self.host_attr} void end(){{")
        if self.target.is_gpu():
            self.print.add_indent(4)

            for p in self.sim.properties:
                pname = p.name()
                tkw = Types.c_keyword(self.sim, Types.Boolean)
                self.print(f"cudaFree(dp_h->{pname}_device_flag_d);")

            self.print(f"cudaFree(dp_d);")
            self.print(f"delete dp_h;")
            self.print(f"delete hp;")

            self.print.add_indent(-4)
        self.print("}")
        self.print("")

    def ifdef_else(self, ifdef, func1, args1, func2, args2):
        self.print.add_indent(4)
        self.print(f"#ifdef {ifdef}")
        func1(*args1)
        self.print("#else")
        func2(*args2)
        self.print("#endif")
        self.print.add_indent(-4)
    
    def generate_ref_name(self, prop, device):
        pname = prop.name()

        if self.target.is_gpu() and device:
            if isinstance(prop, Property):
                return f"dp_d->{pname}_d"
            
            elif isinstance(prop, FeatureProperty):
                return f"{pname}_d"
        else:
            return f"ps->pobj->{pname}"

    def getter_body(self, prop, device=False):
        self.print.add_indent(4)
        tkw = Types.c_accessor_keyword(self.sim, prop.type())
        ptr = self.generate_ref_name(prop, device)

        if isinstance(prop, Property):
            idx = "i"
        elif isinstance(prop, FeatureProperty):
            fname = prop.feature().name()
            idx = f"({prop.feature().nkinds()}*{fname}1 + {fname}2)"

        if Types.is_scalar(prop.type()):
            self.print(f"return {ptr}[{idx}];")
        else:
            nelems = Types.number_of_elements(self.sim, prop.type())
            return_values = [f"{ptr}[{idx}*{nelems} + {n}]" for n in range(nelems)] 
            self.print(f"return {tkw}(" + ", ".join(rv for rv in return_values) + ");")
        self.print.add_indent(-4)

    def get_property(self, prop):
        pname = prop.name()
        tkw = Types.c_accessor_keyword(self.sim, prop.type())

        if isinstance(prop, Property):
            splitname = pname.split('_')
            funcname = ''.join(word.capitalize() for word in splitname)
            params = "const size_t i"

        elif isinstance(prop, FeatureProperty):
            fname = prop.feature().name()
            splitname = fname.split('_') + pname.split('_')
            funcname = ''.join(word.capitalize() for word in splitname)
            params = f"const size_t {fname}1, const size_t {fname}2"
        
        self.print(f"{self.host_device_attr}{tkw} get{funcname}({params}) const{{")

        if self.target.is_gpu() and prop.device_flag:
            self.ifdef_else("__CUDA_ARCH__", self.getter_body, [prop, True], self.getter_body, [prop, False])
        else:
            self.getter_body(prop, False)

        self.print("}")
        self.print("")

    def setter_body(self, prop, device=False):
        self.print.add_indent(4)
        ptr = self.generate_ref_name(prop, device)

        if isinstance(prop, Property):
            idx = "i"
        elif isinstance(prop, FeatureProperty):
            fname = prop.feature().name()
            idx = f"({prop.feature().nkinds()}*{fname}1 + {fname}2)"

        if Types.is_scalar(prop.type()):
            self.print(f"{ptr}[{idx}] = value;")
        else:
            nelems = Types.number_of_elements(self.sim, prop.type())
            for n in range(nelems):
                self.print(f"{ptr}[{idx}*{nelems} + {n}] = value[{n}];")

        if self.target.is_gpu():
            pname = prop.name()
            flag = f"*(dp_d->{pname}_device_flag_d)" if device else f"hp->{pname}_host_flag"
            self.print(f"{flag} = true;")

        self.print.add_indent(-4)

    def set_property(self, prop):
        pname = prop.name()
        tkw = Types.c_accessor_keyword(self.sim, prop.type())

        if isinstance(prop, Property):
            splitname = pname.split('_')
            funcname = ''.join(word.capitalize() for word in splitname)
            self.print(f"{self.host_device_attr}void set{funcname}(const size_t i, const {tkw} &value){{")

        elif isinstance(prop, FeatureProperty):
            fname = prop.feature().name()
            splitname = fname.split('_') + pname.split('_')
            funcname = ''.join(word.capitalize() for word in splitname)
            # Feature properties can only be set from host
            self.print(f"{self.host_attr}void set{funcname}(const size_t {fname}1, const size_t {fname}2, const {tkw} &value){{")

        if self.target.is_gpu():
            if isinstance(prop, Property):
                self.ifdef_else("__CUDA_ARCH__", self.setter_body, [prop, True], self.setter_body, [prop, False])
            
            elif isinstance(prop, FeatureProperty):
                self.setter_body(prop, False)
        else:
            self.setter_body(prop, False)

        self.print("}")
        self.print("")
    
    def sync_ctx_enum(self):
        self.print("enum SyncContext{")
        self.print("    Host = 0,")
        self.print("    Device")
        self.print("};")
        self.print("")

    def sync_property(self, prop):
        pname = prop.name()
        pid = prop.id()
        splitname = pname.split('_')
        funcname = ''.join(word.capitalize() for word in splitname)

        self.print(f"{self.host_attr}void sync{funcname}(SyncContext sync_ctx = Host, bool overwrite = false){{")

        if self.target.is_gpu():
            self.print.add_indent(4)
            self.print(f"cudaMemcpy(&(hp->{pname}_device_flag_h), dp_h->{pname}_device_flag_d, sizeof(bool), cudaMemcpyDeviceToHost);")
            self.print("")
            
            #####################################################################################################################
            #####################################################################################################################

            self.print(f"if (hp->{pname}_host_flag && hp->{pname}_device_flag_h){{")
            self.print(f"    PAIRS_ERROR(\"OUT OF SYNC 1! Both host and device versions of {pname} are in a modified state.\\n\");")
            self.print("    exit(-1);")
            self.print("}")
            self.print(f"else if(sync_ctx==Host && overwrite==false){{")
            self.print(f"   if (hp->{pname}_host_flag && !ps->pairs_runtime->getPropFlags()->isHostFlagSet({pid})){{")
            self.print(f"       PAIRS_ERROR(\"OUT OF SYNC 2! Did you forget to sync{funcname}(Host) before calling set{funcname} from host? Use sync{funcname}(Host,true) if you want to overwrite {pname} values in host.\\n\");")
            self.print("        exit(-1);")
            self.print("    }")
            self.print("}")
            self.print(f"else if(sync_ctx==Device && overwrite==false){{")
            self.print(f"   if (hp->{pname}_device_flag_h && !ps->pairs_runtime->getPropFlags()->isDeviceFlagSet({pid})){{")
            self.print(f"       PAIRS_ERROR(\"OUT OF SYNC 3! Did you forget to sync{funcname}(Device) before calling set{funcname} from device? Use sync{funcname}(Device,true) if you want to overwrite {pname} values in device.\\n\");")
            self.print("        exit(-1);")
            self.print("    }")
            self.print("}")
            self.print("")

            #####################################################################################################################
            #####################################################################################################################

            self.print(f"if (hp->{pname}_host_flag){{")
            self.print(f"    ps->pairs_runtime->getPropFlags()->setHostFlag({pid});")
            self.print(f"    ps->pairs_runtime->getPropFlags()->clearDeviceFlag({pid});")
            self.print("}")
            
            self.print(f"else if (hp->{pname}_device_flag_h){{")
            self.print(f"    ps->pairs_runtime->getPropFlags()->setDeviceFlag({pid});")
            self.print(f"    ps->pairs_runtime->getPropFlags()->clearHostFlag({pid});")
            self.print("}")
            self.print("")

            nelems = Types.number_of_elements(self.sim, prop.type())
            tkw = Types.c_keyword(self.sim, prop.type())

            self.print(f"if (sync_ctx==Device) {{")
            self.print(f"    ps->pairs_runtime->copyPropertyToDevice({pid}, ReadOnly, (((ps->pobj->nlocal + ps->pobj->nghost) * {nelems}) * sizeof({tkw})));")
            self.print("}")
            self.print("")

            self.print(f"if (sync_ctx==Host) {{")
            self.print(f"    ps->pairs_runtime->copyPropertyToHost({pid}, ReadOnly, (((ps->pobj->nlocal + ps->pobj->nghost) * {nelems}) * sizeof({tkw})));")
            self.print("}")
            self.print("")

            self.print(f"hp->{pname}_host_flag = false;")
            self.print(f"hp->{pname}_device_flag_h = false;")
            self.print(f"cudaMemcpy(dp_h->{pname}_device_flag_d, &(hp->{pname}_device_flag_h), sizeof(bool), cudaMemcpyHostToDevice);")

            self.print.add_indent(-4)
        self.print("}")
        self.print("")
        
    def sync_feature_property(self, fp):
        fp_id = fp.id()
        fp_name = fp.name()
        f_name = fp.feature().name()
        splitname = f_name.split('_') + fp_name.split('_')
        funcname = ''.join(word.capitalize() for word in splitname)

        self.print(f"{self.host_attr}void sync{funcname}(SyncContext sync_ctx = Host){{")

        if self.target.is_gpu():
            self.print.add_indent(4)
            self.print(f"if (hp->{fp_name}_host_flag && sync_ctx==Device) {{")
            self.print(f"    ps->pairs_runtime->copyFeaturePropertyToDevice({fp_id});")
            self.print("}")
            self.print("")

            self.print(f"hp->{fp_name}_host_flag = false;")
            self.print.add_indent(-4)

        self.print("}")
        self.print("")

    def utility_funcs(self):
        nlocal = "ps->pobj->nlocal"
        nlocal_d = "dp_d->nlocal"
        nghost = "ps->pobj->nghost"
        nghost_d = "dp_d->nghost"

        if self.target.is_gpu():
            self.print(f"{self.host_device_attr}int size() const {{")
            self.print(f"    #ifdef __CUDA_ARCH__")
            self.print(f"        return {nlocal_d} + {nghost_d};")
            self.print(f"    #else")
            self.print(f"        return {nlocal} + {nghost};")
            self.print(f"    #endif")
            self.print("}")
            self.print("")
        else:
            self.print(f"int size() const {{return {nlocal} + {nghost};}}")

        if self.target.is_gpu():
            self.print(f"{self.host_device_attr}int nlocal() const {{")
            self.print(f"    #ifdef __CUDA_ARCH__")
            self.print(f"        return {nlocal_d};")
            self.print(f"    #else")
            self.print(f"        return {nlocal};")
            self.print(f"    #endif")
            self.print("}")
            self.print("")
        else:
            self.print(f"int nlocal() const {{return {nlocal};}}")

        if self.target.is_gpu():
            self.print(f"{self.host_device_attr}int nghost() const {{")
            self.print(f"    #ifdef __CUDA_ARCH__")
            self.print(f"        return {nghost_d};")
            self.print(f"    #else")
            self.print(f"        return {nghost};")
            self.print(f"    #endif")
            self.print("}")
            self.print("")
        else:
            self.print(f"int nghost() const {{return {nghost};}}")


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
