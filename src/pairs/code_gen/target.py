class Target:
    # Architectures
    Arch_x86 = 0
    Arch_Nvidia = 1

    # Backend
    Backend_CPP = 0
    Backend_CUDA = 1
    Backend_LLVM = 2

    # Features
    Feature_CPU = 1
    Feature_AVX = 2
    Feature_AVX2 = 3
    Feature_AVX512 = 4
    Feature_GPU = 5
    Feature_OpenMP = 6

    # Operating system
    OS_Unknown = 0
    OS_Linux = 1
    OS_Windows = 2

    def __init__(self, backend, features, arch=None, os=None):
        self.backend = backend
        self.features = features if isinstance(features, list) else [features]
        self.arch = arch if arch is not None else Target.Arch_x86 if Target.Feature_CPU in self.features else Target.Arch_Nvidia
        self.os = os if os is not None else Target.OS_Unknown

    def add_feature(self, feature):
        self.features.append(feature)

    def has_feature(self, feature):
        return feature in self.features

    def is_cpu(self):
        return self.has_feature(Target.Feature_CPU)

    def is_gpu(self):
        return self.has_feature(Target.Feature_GPU)

    def is_openmp(self):
        return self.has_feature(Target.Feature_OpenMP)
