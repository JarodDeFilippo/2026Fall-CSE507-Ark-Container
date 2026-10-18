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

`compare_cyclic_runs.py` also takes `--models teacher|student|both` (default
`both`: a dashed student and a solid teacher line per run; `teacher` or
`student` draws that series only, solid, with the run label alone in the
legend) and `--full-length` (draw every run to its own last evaluated cycle
instead of cutting all runs at the shortest run's last cycle; the title then
drops "through cycle N"). `--colors RUN=#RRGGBB ...` and `--labels RUN=Label
...` set a line color and a legend label per run, keyed by the run directory
name: runs that are not named keep the default palette color and short label,
names that match no compared run are ignored with a note, a label with spaces
needs quotes, and the output folder name does not change. Give them after the
run directories, or end the option list with `--`. Plotted mAUC is in percent
(the CSV fractions times 100) and the y axis reads "Test mAUC (%)" or
"Validation mAUC (%)", so `--y-min` and `--y-max` are in percent in both
scripts.
