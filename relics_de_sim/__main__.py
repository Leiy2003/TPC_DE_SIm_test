"""Tiny dispatch CLI: ``python -m relics_de_sim {de,cevns,acceptance,...}``.

Each subcommand defers to the corresponding module under ``scripts/`` so the
real argparse parsers stay near their implementations.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m relics_de_sim {de,cevns,acceptance,waveform,fit_cuts}", file=sys.stderr)
        sys.exit(2)

    cmd = sys.argv.pop(1)
    if cmd == "de":
        from scripts.run_de_sim import main as run

        run()
    elif cmd == "cevns":
        from scripts.run_cevns_sim import main as run

        run()
    elif cmd == "acceptance":
        from scripts.run_acceptance import main as run

        run()
    elif cmd == "waveform":
        from scripts.run_waveform_gen import main as run

        run()
    elif cmd == "fit_cuts":
        from scripts.fit_cuts import main as run

        run()
    else:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
