#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/gdec
# All rights reserved.
#

import os
import io
import json
from pathlib import Path
from argparse import ArgumentParser
from contextlib import redirect_stdout

import torch
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_DoG
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, train_test_exp, renderer=render_DoG):
    root = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(root, "renders")
    gt_path = os.path.join(root, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    if not views:
        raise ValueError(f"No {name} cameras to evaluate")
    for _ in range(5):
        renderer(views[0], gaussians, pipeline, background, use_trained_exp=train_test_exp)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for index, view in enumerate(views):
        start.record()
        rendering = renderer(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
        )["render"]
        end.record()
        end.synchronize()
        elapsed_ms += start.elapsed_time(end)
        gt = view.original_image[:3]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        filename = f"{index:05d}.png"
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, filename))
        torchvision.utils.save_image(gt, os.path.join(gt_path, filename))

    (Path(root) / "render_timing.json").write_text(json.dumps({
        "views": len(views), "mean_render_ms": elapsed_ms / len(views)
    }, indent=2))


def render_sets(dataset, iteration, pipeline, skip_train, skip_test):
    with torch.no_grad():
        if pipeline.compute_cov3D_python:
            raise ValueError(
                "DoG rendering requires CUDA covariance computation.")

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset, gaussians, load_iteration=iteration, shuffle=False)
        background = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

        summary = Path(dataset.model_path) / "training_summary.json"
        renderer = render_DoG
        if summary.exists() and scene.loaded_iter < json.loads(summary.read_text())["dog_start_iter"]:
            renderer = render
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                renderer,
            )
        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                renderer,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="DoG rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    with redirect_stdout(io.StringIO()):
        args = get_combined_args(parser)
        safe_state(True)
        render_sets(
            model.extract(args),
            args.iteration,
            pipeline.extract(args),
            args.skip_train,
            args.skip_test,
        )
