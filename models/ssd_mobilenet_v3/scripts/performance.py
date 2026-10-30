"""Reserved standalone whole-model benchmark entry point; native qualification is pending.

The model intentionally has no performance declaration or invented throughput
expectation. An MLA-only benchmark must not masquerade as whole-model speed.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.parse_args()
    raise SystemExit("SSD-MobileNet V3 whole-model native performance is not qualified; no report was produced")


if __name__ == "__main__":
    main()
