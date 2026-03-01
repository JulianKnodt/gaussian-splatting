import os
from argparse import ArgumentParser
import torch
import numpy as np
import json
import time

from arguments import OptimizationParams, PipelineParams, ModelParams

import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv

from scene import Scene, GaussianModel
from scene.cameras import MiniCam2, lookAt
from gaussian_renderer import render

from tqdm import trange
from utils.loss_utils import ssim

import matplotlib.pyplot as plt

def arguments():
  a = ArgumentParser()
  a.add_argument("--target", required=True)
  a.add_argument("--init", required=True)
  a.add_argument("--iters", default=10000)
  a.add_argument("--output", required=True)
  a.add_argument("--stats", help="Stat file")
  a.add_argument("--from-center", action="store_true", help="Set the eye at the origin")
  a.add_argument("--from-upper-hemi", action="store_true", help="Only optimize upper hemisphere")
  a.add_argument(
    "-lr", "--learning-rate", default=3e-3, type=float, help="Overall learning rate"
  )
  a.add_argument("--radius", default=8., type=float, help="Radius of camera")
  lp = ModelParams(a)
  op = OptimizationParams(a)
  pp = PipelineParams(a)
  return a.parse_args(), lp, op, pp

def main():
  args, lp, op, pp = arguments()
  print(f"[INFO]: Optimizing {args.init} to {args.target}")
  lp = lp.extract(args)
  op = op.extract(args)
  pipe = pp.extract(args)
  init = GaussianModel(3, op.optimizer_type)
  init.load_ply(args.init)

  lr = args.learning_rate
  params = [
    {'params': [init._features_dc], 'lr': lr},
    {'params': [init._features_rest], 'lr': lr/15},
    {'params': [init._opacity], 'lr': 2.5e-2},

    #init._scaling,
    {'params': [init._scaling], 'lr': lr/2},
    {'params': [init._rotation], 'lr': lr/5},
    {'params': [init._xyz], 'lr': lr/20},
  ]
  opt = optim.Adam(params, lr=lr)

  target = GaussianModel(3, "default")
  target.load_ply(args.target)

  min_vals = target._xyz.min(dim=0).values
  max_vals = target._xyz.max(dim=0).values
  mid = (max_vals + min_vals) / 2
  if args.from_center: mid = 0
  scale = 2 / torch.max(max_vals - min_vals)
  if args.from_center: scale *= 32

  target._xyz.data -= mid
  target._xyz.data *= scale
  target._scaling.data += scale.log()

  init._xyz.data -= mid
  init._xyz.data *= scale
  init._scaling.data += scale.log()

  if args.from_center: args.iters *= 2

  start = time.time()
  t = trange(args.iters)
  losses = []
  for it in t:
    opt.zero_grad()

    rand_dir = np.random.randn(3)
    rand_dir = rand_dir / np.linalg.norm(rand_dir)
    rand_dir *= args.radius

    if args.from_upper_hemi: rand_dir[1] = rand_dir[1].abs()

    if not args.from_center:
      look_at_mat = lookAt(rand_dir, [0,0,0], [0, -1, 0])
    else:
      p = (np.random.random(3) - 0.5) * 5.
      look_at_mat = lookAt(p, -rand_dir/8., [0, -1, 0])

    viewpoint_cam = MiniCam2(
      1024, 1024,
      6, 6,
      1e-2, 100,
      look_at_mat,
    )

    bg = torch.rand((3), device="cuda")
    render_pkg = render(viewpoint_cam, init, pipe, bg, use_trained_exp=False, separate_sh=False)
    got_img = render_pkg["render"]
    with torch.no_grad():
      exp_render_pkg = render(viewpoint_cam, target, pipe, bg, use_trained_exp=False, separate_sh=False)
      exp_img = exp_render_pkg["render"]

    loss = F.l1_loss(got_img, exp_img)

    if it % 500 == 0:
      tv.utils.save_image(exp_img, "tmp_exp.png")
      tv.utils.save_image(got_img, "tmp_got.png")
    loss.backward()

    opt.step()
    t.set_postfix(L=f"{loss.item():.03e}")
    losses.append(loss.item())

  plt.plot(losses, label="losses")
  plt.savefig("tmp.pdf")

  elapsed = time.time() - start

  init._scaling.data -= scale.log()
  init._xyz.data /= scale
  init._xyz.data += mid

  print(f"[INFO]: Saved {args.init} to {args.output}")
  init.save_ply(args.output)
  if args.stats:
    with open(args.stats, "w") as s:
      data = {
        "elapsed_ms": elapsed,
        # todo quality
      }
      json.dump(data, s, indent=2)


if __name__ == "__main__": main()
