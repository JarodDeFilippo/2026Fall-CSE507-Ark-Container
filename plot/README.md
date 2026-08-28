# Ark+ run plots

`plot_cyclic_run.py` reads the evaluation CSVs produced beneath an Ark+ run's
`evaluation/<dataset>/` directory.

From the repository root, run:

~~~bash
python plot/plot_cyclic_run.py experiment_seed_100
~~~

By default, the script plots the mAUC rows and writes figures beneath
`plot/<run_name>/` in three plot-type directories. Each plot type is divided
again into validation/ and test/:

- focused/split/: each dataset figure has solid student and teacher lines
  using evaluations immediately after that dataset was trained.
- unfocused/split/: each dataset figure has faint dashed student and teacher
  lines using every performance measurement for the target dataset, including
  evaluations after training the target dataset itself.
- focused_unfocused/split/: each dataset figure overlays the two focused
  solid lines on the two complete dashed trajectories for the same split.

Useful options:

~~~text
--metric mAUC
--format png|pdf|svg
--dpi 180
--y-min VALUE
--y-max VALUE
--output-root DIRECTORY
~~~

# Comparing multiple runs

`compare_cyclic_runs.py` compares all shared datasets across two or more Ark+
runs.

From the repository root, run:

~~~bash
python plot/compare_cyclic_runs.py \
    experiment_seed_100 \
    experiment_seed_101
~~~
