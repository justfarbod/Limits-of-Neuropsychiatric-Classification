import argparse
import json

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Maximum-entropy analysis of connectivity representations"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in [
        "validate-inputs",
        "build-fc",
        "empirical",
        "select",
        "fit-source",
        "synthetic",
        "mechanical",
        "plot",
    ]:
        subcommands.add_parser(command).add_argument(
            "--config", required=True, help="External JSON configuration"
        )
    demo = subcommands.add_parser("demo")
    demo.add_argument(
        "--output-dir",
        help="External directory; otherwise a temporary directory is created",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo

            run_demo(args.output_dir)
        else:
            config = load_config(args.config)
            if args.command == "validate-inputs":
                from .inputs import load_inputs
                from .targets import build_targets

                data = load_inputs(config)
                targets = build_targets(data.metadata, config["targets"])
                print(
                    json.dumps(
                        {
                            "rows": len(data.features),
                            "features": data.features.shape[1],
                            "outcomes": len(targets),
                        }
                    )
                )
            elif args.command == "build-fc":
                from .inputs import build_features

                build_features(config)
            elif args.command == "empirical":
                from .empirical.workflow import run_empirical

                run_empirical(config)
            elif args.command == "select":
                from .selection.workflow import run_selection

                run_selection(config)
            elif args.command == "fit-source":
                from .synthetic.source import fit_source

                fit_source(config)
            elif args.command == "synthetic":
                from .synthetic.workflow import run_synthetic

                run_synthetic(config)
            elif args.command == "mechanical":
                from .mechanical.quadrature import run_mechanical

                run_mechanical(config)
            elif args.command == "plot":
                from .reporting.plots import plot_results

                plot_results(config)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as error:
        # Third-party exceptions can include private paths or row values.
        parser.exit(
            2,
            f"{type(error).__name__}: analysis could not complete; check the external configuration and inputs.\n",
        )
    print("Completed.")


if __name__ == "__main__":
    main()
