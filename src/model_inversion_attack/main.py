from model_inversion_attack import enroll, mi_attack, server, reconstruct
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Biometric Privacy Research Framework"
    )

    parser.add_argument(
        "mode",
        choices=["server", "attack", "enroll", "reconstruct"],
        help="Operation mode"
    )
    parser.add_argument(
        "--input",
        default=None
    )
    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    try:
        if args.mode == "server":
            server.run_server(args)

        elif args.mode == "attack":
            mi_attack.run_attack(args)

        elif args.mode == "enroll":
            enroll.run_enroll(args)

        elif args.mode == "reconstruct":
            reconstruct.run_reconstruct(args)

    except KeyboardInterrupt:
        print("\033[?25h\n[INFO] Interrupted by user (Ctrl+C). Shutting down gracefully...")
        sys.exit(0)

    except Exception as e:
        print(f"\033[?25h\n[ERROR] Unexpected failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
