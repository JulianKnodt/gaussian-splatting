import os
import argparse
import sys
import time
import json
from itertools import chain

if os.name == "nt":
  bin_file = f"{sys.executable} train_to_match.py"
else:
  bin_file = "tmp"

args = None

abl_dir = "ablations"
out_dir = "outputs"
output_sub_dir = lambda sd: os.path.join(out_dir, sd)

def run(
  src, dst, flags, out_dir=abl_dir, src_dir="data", bin=bin_file, eval=True,
  missing_only=False, stage="run", input_flag="--init", output_flag="--output",
):
  assert(bin != ...), "Must specify bin for now"
  def cb():
    nonlocal missing_only
    if stage not in args.stages: return []
    if args.match_output is not None and args.match_output not in dst: return []
    if args.force: missing_only = False
    out_json = f"{out_dir}/{dst[:-4]}.json"
    if missing_only and os.path.exists(out_json) and not args.force:
      print(f"[⏭️]: Skipping {src} -> {dst}, results at {out_json} already exist")
      return []
    out_file = f"{out_dir}/{dst}"
    if missing_only and (not eval) and os.path.exists(out_file):
      print(f"[⏭️]: Skipping {src} -> {out_file}, destination exists")
      return []
    cmds = []
    if not args.eval_only:
      cmds.append(f"{bin} {input_flag} {src_dir}/{src} {output_flag} {out_dir}/{dst} {flags} --stats {out_json}")
    if eval and out_file.endswith(".obj"):
      cmds.append(
        f"{sys.executable} bin/hausdorff.py -o {src_dir}/{src} -n {out_file} --stat-file {out_json}"
      )
    return cmds
  return cb

def render(
  i, cy,cz, ly,lz,out, w=1024, h=1024, cx=0, fy=0, lx=0, rz=45,
  extras="",
  missing_only=False,
):
  out = os.path.join(os.getcwd(), out)
  def cb():
    if "render" not in args.stages: return []
    if args.match_output is not None and args.match_output not in out: return []
    if (not args.force) and missing_only and os.path.exists(out): return []

    cmd = f"{sys.executable} bin/render.py \
      --mesh {i} \
      --cam-x {cx} --rot-z {rz} \
      --cam-y {cy} --cam-z {cz} --lookat-y {ly} --lookat-z {lz} \
      -o {out} --width {w} --floor-y {fy} --lookat-x {lx} --height {h} {extras} "
    if not args.debug_render: cmd += " --final-render --samples 256"
    return [cmd]
  return cb

def runnable_cmds(cmds, stage_kind="run"):
  def cb():
    if stage_kind not in args.stages: return []
    missing_only = "" if not args.missing_only else " --missing-only "
    return [ c + missing_only for c in cmds ]
  return cb

data_dir = "../../rust/gaussian_qem/data"
data = [
  f for f in os.listdir(data_dir)
  if "ply" in f and "hotel" not in f
]

center_eye = {
  "playroom.ply",
  "cathedral.ply",
}

experiments = {
  #"run-all": [ "run-50", "run-25", "run-10", ],
  "run-50": [
    run(
      f"{m[:-4]}_50.ply",
      f"{m[:-4]}_50.ply",
      f"--target {data_dir}/{m} {'--from-center' if m in center_eye else ''}",
      src_dir="../../rust/gaussian_qem/outputs",
      out_dir=out_dir,
      missing_only=True,
    )
    for m in data
  ],
  "run-25": [
    run(
      f"{m[:-4]}_25.ply",
      f"{m[:-4]}_25.ply",
      f"--target {data_dir}/{m} {'--from-center' if m in center_eye else ''}",
      src_dir="../../rust/gaussian_qem/outputs",
      out_dir=out_dir,
      missing_only=True,
    )
    for m in data
  ],
  "run-10": [
    run(
      f"{m[:-4]}_10.ply",
      f"{m[:-4]}_10.ply",
      f"--target {data_dir}/{m} {'--from-center' if m in center_eye else ''}",
      src_dir="../../rust/gaussian_qem/outputs",
      out_dir=out_dir,
      missing_only=True,
    )
    for m in data
  ],
}

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument(
    "-e", "--experiments",
    default=list(experiments.keys()),
    nargs="*",
    choices=list(experiments.keys()),
  )
  a.add_argument(
    "--stages", default=["run", "render", "plot"], nargs="+", choices=["run", "render", "plot"],
    help="What steps of testing to run"
  )
  a.add_argument(
    "--debug-render", action="store_true", help="Faster debug render instead of final version"
  )
  a.add_argument("--missing-only", action="store_true", help="Run complete dataset for only missing files")
  a.add_argument("--first-only", action="store_true", help="Run one command then exit")
  a.add_argument("--skip-to", default=None, choices=list(experiments.keys()), help="skip to this experiment")
  a.add_argument("--match-output", default=None, help="Only match render outputs with this")
  a.add_argument("--force", action="store_true", help="Run all commands even if marked missing only")
  a.add_argument("--eval-only", action="store_true", help="Only run evaluation")
  a.add_argument("--dry-run", action="store_true", help="Print commands and exit")
  return a.parse_args()

args = arguments()
now = time.asctime(time.localtime())

experiment_timestamps = {}

exp_file = "experiment_log.json"

def run_experiment(exp):
  global experiment_timestamps
  print("-================================================-")
  print(f"\tStarting {exp}")
  print("-================================================-")
  if os.path.exists(exp_file):
    with open(exp_file, "r") as f:
      experiment_timestamps = json.load(f)

  for cmd_list in experiments[exp]:
    # run sub-experiment if it's a string
    if type(cmd_list) == str:
      run_experiment(cmd_list)
      continue

    for cmd in cmd_list():
      if args.dry_run:
        print(cmd)
        continue
      assert(not os.system(cmd)), cmd
      if args.first_only: exit()
  print("-================================================-")
  print(f"\tFinished {exp}!")
  print("-================================================-")
  experiment_timestamps[exp] = {
    "time": now,
    "os": os.name,
  }

  if args.dry_run: return

  # write after finishing each experiment so that even if stopped halfway then it will stop.
  with open(exp_file, "w") as f:
    json.dump(experiment_timestamps, f, indent=2)

for exp in args.experiments:
  if args.skip_to is not None:
    if args.skip_to == exp: args.skip_to = None
    else: continue
  run_experiment(exp)
