"""Terminal wrapper: python -m openkapitza.agent [--answers FILE] [--potentials DIR]

Reads lines, hands them to Session, prints the replies. With --answers, the
lines come from a file (one per prompt; blank line = take the default),
echoed as they are consumed -- that is how the scripted cases run.
"""

from __future__ import annotations

import argparse
import sys

from .session import Session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m openkapitza.agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", help="file of answers, one per line ('#' lines ignored)")
    ap.add_argument("--potentials", help="LAMMPS potentials folder (for 'prepare jobs')")
    ap.add_argument("--samples", type=int, default=1000, help="Monte Carlo samples")
    a = ap.parse_args(argv)
    s = Session(potentials=a.potentials, n_samples=a.samples)
    print(s.start())
    if a.answers:
        with open(a.answers, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if not ln.lstrip().startswith("#")]
        for ln in lines:
            print(f"\n> {ln}")
            print(s.handle(ln))
            if s.done:
                break
        return 0
    while not s.done:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        print(s.handle(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
