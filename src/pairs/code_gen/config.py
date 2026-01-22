import argparse
from pairs.code_gen.target import Target

class PairsConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Code generator for P4IRS"
        )

        parser.add_argument(
            "--interface-name",
            required=True,
            help="Name of the interface library and the generated header file"
        )
        parser.add_argument(
            "--target",
            required=True,
            choices=["cpu", "gpu"],
            help="Target architecture (cpu or gpu)"
        )

        parser.add_argument(
            "--output-dir",
            required=True,
            help="Directory where generated files will be written"
        )

        parser.add_argument(
            "--debug",
            required=False,
            type=int,
            choices=[0,1],
            default=0,
            help="Enable debug mode"
        )

        args = parser.parse_args()

        self.ref = args.interface_name
        self.output_dir = args.output_dir
        self.debug = bool(args.debug)

        if args.target == 'gpu':
            self.target = Target(Target.Backend_CUDA, Target.Feature_GPU)
        elif args.target == 'cpu':
            self.target = Target(Target.Backend_CPP, Target.Feature_CPU)

