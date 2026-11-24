# Safe on every leg

Code and results for the paper *Safe on every leg: a multi-objective multi-drop container loading problem with en-route axle and rollover margins* (manuscript in preparation).

In multi-drop delivery the cargo on board changes at every stop, so a load that is legal at the depot need not stay legal on later legs. This repository holds a three-objective container loading model with utilisation, unloading obstacles and the worst safety margin over all legs of the route; an exact grid model for small instances; a random-key block-building decoder with four evolutionary searches; and every experiment of the paper with its results.

## Contents

| Folder | What it holds |
|---|---|
| `code/legsafe/` | The model: instances (`instances.py`), vehicle (`vehicle.py`), leg loads and margins (`legs.py`), objectives, decoder, NSGA-II (`nsga2.py`), NSGA-III, MOEA/D, SMS-EMOA and the quality indicators (`moea.py`), the exact grid model (`exact.py`). |
| `code/experiments/` | One script per experiment, its summary, and `paper.py`, which writes the tables and figures of the paper. |
| `code/tests/` | 447 tests, among them checks that the compiled kernels return exactly what the plain reference code returns. |
| `results/` | The result file of every experiment, one row per plan or per checkpoint. |
| `data/` | Empty in the repository; see below. |

## Setup

Python 3.11 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The benchmark instances of Bischoff and Ratcliff are not part of the repository. The following command downloads them from the OR-Library (classes BR1 to BR7) and from the OR-Datasets collection (classes BR8 to BR15) and checks them against the files the experiments were run on:

```
cd code
python -m experiments.get_data
```

All further commands are run from the `code` folder. The tests need the instance files:

```
python -m pytest -q
```

## Reproducing the paper

Runs are seeded, so each command below writes the result file in `results/` that the paper uses. Add `--workers N` to run N searches at a time.

| Section of the paper | Command | Result file |
|---|---|---|
| Exact fronts on small instances (E1) | `python -m experiments.e1_exact exact`, then `heuristic` and `extended`; `python -m experiments.e1_reach` and `python -m experiments.e1_reach --extended --limit 3e6` | `e1_*.csv` |
| Budget and convergence | `python -m experiments.convergence --grid types` | `convergence_types.csv` |
| The same with the search variants that were tried | add `--variant distinct`, `one_key`, `seeded --up-to 4000` or `two_ends --up-to 4000`, or `--classes 10 15 --stops 5 10 20` | `convergence_types_*.csv` |
| Plain loading problem | `python -m experiments.plain --classes 1 2 3 4 5 6 7 --instances 10 --budgets 1000 2000 5000 --out plain.csv` and `python -m experiments.plain --classes 8 9 10 11 12 13 14 15 --instances 5 --budgets 1000 2000 --out plain_large.csv` | `plain.csv`, `plain_large.csv` |
| Departure balance and the cost of safety (E2, E3, E5) | `python -m experiments.final --name main --classes 1 2 3 4 5 6 7 --instances 20 22 --runs 5 --objectives every_leg departure centre` | `final_main.csv.gz` |
| Five stops and rollover | `python -m experiments.final --name stops --classes 3 4 5 6 7 --instances 20 22 --stops 5 --runs 2` | `final_stops.csv` |
| Search algorithms (E4) | `python -m experiments.final --name algorithms --classes 1 3 5 7 --instances 20 21 --weights 0 2 --runs 10 --algorithms nsga2 nsga3 moead sms_emoa` | `final_algorithms.csv.gz` |

The summaries print every number quoted in the paper:

```
python -m experiments.e1_summary
python -m experiments.convergence_summary --grid types
python -m experiments.final_summary objectives main
python -m experiments.final_summary stops stops
python -m experiments.final_summary algorithms algorithms
python -m experiments.plain_summary
```

`python -m experiments.paper` writes the tables and figures of the paper into `manuscript/generated/` and `manuscript/figures/`.

The two final files were written in two parts, the first runs and then the later ones (`--first-run`), and merged. A run's seed depends only on its case and its run number, so a single command as above gives the same file. Files ending in `.csv.gz` are read directly by the summaries.

Scripts named `*_pilot.py`, `tune_*.py` and `timing.py`, and the result files `e2_pilot.csv`, `e5_*.csv`, `tune_*.csv` and `convergence.csv`, belong to the preliminary studies that set up the final experiments.

## Assumptions to keep in mind

- The reference vehicle is the rigid IVECO Eurocargo of the worked example of Krebs and Ehmke (2021). Its kerb weights come from the manufacturer's description of a sister variant, the mass of its body is inferred, and the two heights used by the rollover margin are assumed (`code/experiments/common.py`).
- Box weights are generated as described by Elahi et al. (2026); the payload (10.1 t) and the target utilisation (0.9) of the generator are our choice.

## Licence

To be chosen by the authors before the repository is made public.
