"""Background worker: claims jobs from queue.txt in order and runs them, within a time budget.

  python run_queue.py <budget_minutes>
A job is claimed by creating runs/<job>.claim atomically; finished jobs have runs/<job>/info.json.
A job starts only if its estimated duration fits in the remaining budget.
"""
import os, subprocess, sys, time
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

W, OUT = HERE, RUNS
EST = {"real": 10, "combined": 22, "cnn": 5, "cnn_lstm": 15, "cnn_gru": 14, "cldnn": 20, "bilstm": 4,
       "transformer": 4}


def job_name(a):
    if a[0] == "syn":
        return f"syn_{a[1]}_s{a[2]}" + ("" if len(a) < 4 or a[3] == "2" else f"_{a[3]}cls")
    ncls = a[4] if len(a) > 4 else "2"
    cfo = float(a[5]) if len(a) > 5 else 0.0
    return f"hw_{a[1]}_{a[2]}_s{a[3]}" + ("" if ncls == "2" else f"_{ncls}cls") + (f"_cfo{cfo:g}k" if cfo else "")


def estimate(a):
    return EST[a[2]] if a[0] == "hw" else EST[a[1]]


def main():
    budget = float(sys.argv[1]) * 60
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    while True:
        started = False
        for line in open(f"{W}/queue.txt"):
            a = line.split()
            if not a or a[0].startswith("#"):
                continue
            job = job_name(a)
            if os.path.exists(f"{OUT}/{job}/info.json"):
                continue
            if time.time() - t0 + estimate(a) * 60 > budget:
                continue
            try:
                os.mkdir(f"{OUT}/{job}.claim")
            except FileExistsError:
                continue
            print(f"[{time.strftime('%H:%M:%S')}] start {job}", flush=True)
            r = subprocess.run([sys.executable, "-I", f"{W}/run_experiment.py", *a],
                               env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
            print(f"[{time.strftime('%H:%M:%S')}] end {job} rc={r.returncode} "
                  f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
            started = True
            break
        if not started:
            print(f"[{time.strftime('%H:%M:%S')}] nothing left that fits the budget; exiting", flush=True)
            return


if __name__ == "__main__":
    main()
