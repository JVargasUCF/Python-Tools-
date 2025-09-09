"""
Unified interface for torch wall analysis tools.

This script exposes two GUI entry points:
- run_data_gui(): wrapper for the GUI in ``Torch Calcer`` which analyzes temperature
  point datasets and computes wall heat fluxes.
- run_model_gui(): wrapper for the GUI in ``torch_wall_analyzer`` which predicts
  the wall temperature profile using a simple engineering model.

Usage::

    python torch_combined.py --mode data   # launch temperature data analyzer
    python torch_combined.py --mode model  # launch torch wall analyzer
"""

from importlib.machinery import SourceFileLoader
import argparse

# Load the original scripts whose filenames are not valid module names.
_tc_loader = SourceFileLoader("torch_calcer", "Torch Calcer")
torch_calcer = _tc_loader.load_module()

_twa_loader = SourceFileLoader("torch_wall_analyzer", "torch_wall_analyzer")
torch_wall_analyzer = _twa_loader.load_module()


def run_data_gui():
    """Launch the GUI from ``Torch Calcer`` for temperature point analysis."""
    torch_calcer.run_gui()


def run_model_gui():
    """Launch the GUI from ``torch_wall_analyzer`` for model-based analysis."""
    torch_wall_analyzer.run_gui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified entry point for torch wall analysis tools."
    )
    parser.add_argument(
        "--mode",
        choices=["data", "model"],
        default="data",
        help=(
            "Select which GUI to launch: 'data' for Torch Calcer or "
            "'model' for torch_wall_analyzer."
        ),
    )
    args = parser.parse_args()

    if args.mode == "data":
        run_data_gui()
    else:
        run_model_gui()
