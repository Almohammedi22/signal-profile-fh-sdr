"""Folder locations for the replication code. Each can be overridden with an environment variable."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SPR_DATA", os.path.join(HERE, "data"))            # unzipped release files
CACHE = os.environ.get("SPR_CACHE", os.path.join(HERE, "work", "cache"))  # written by prepare_cache.py
RUNS = os.environ.get("SPR_RUNS", os.path.join(HERE, "work", "runs"))     # one folder per training run
RESULTS = os.environ.get("SPR_RESULTS", os.path.join(HERE, "..", "results"))
FIGURES = os.environ.get("SPR_FIGURES", os.path.join(HERE, "work", "figures"))
for _d in (CACHE, RUNS, RESULTS, os.path.join(RESULTS, "paper"), FIGURES):
    os.makedirs(_d, exist_ok=True)
