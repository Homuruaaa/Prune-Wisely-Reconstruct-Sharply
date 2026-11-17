#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import json

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def detail_errors(pred, gt):
    weights = gt.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    def luminance(rgb):
        return F.pad((rgb * weights).sum(1, keepdim=True), (1, 1, 1, 1), mode="replicate")

    sobel = gt.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernels = torch.stack((sobel, sobel.T)).unsqueeze(1)
    gt_luma = luminance(gt)
    gradient = F.conv2d(gt_luma, kernels)
    mask = gradient.square().sum(dim=1, keepdim=True) >= 0.05 ** 2
    error = (pred - gt).abs().mean(dim=1, keepdim=True)
    edge_mae = error[mask].mean().item() if mask.any() else None
    laplacian = gt.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    detail = F.conv2d(luminance(pred) - gt_luma, laplacian)
    return edge_mae, detail.abs().mean().item()


def load_image(path):
    with Image.open(path) as image:
        return tf.to_tensor(image.convert("RGB")).unsqueeze(0).cuda()


@torch.no_grad()
def evaluate(model_paths):
    perceptual = LPIPS("vgg").cuda().eval()
    for scene_dir in map(Path, model_paths):
        averages, per_view = {}, {}
        for method in sorted((scene_dir / "test").iterdir()):
            if not method.is_dir():
                continue
            renders = sorted((method / "renders").glob("*.png"))
            if not renders or {p.name for p in renders} != {p.name for p in (method / "gt").glob("*.png")}:
                raise ValueError(f"Empty or unmatched test images in {method}")
            scores = {}
            for path in tqdm(renders, desc=method.name):
                pred, gt = load_image(path), load_image(method / "gt" / path.name)
                edge, detail = detail_errors(pred, gt)
                values = dict(PSNR=psnr(pred, gt).item(), SSIM=ssim(pred, gt).item(),
                              LPIPS=perceptual(pred, gt).item(), edge_mae=edge,
                              laplacian_mae=detail)
                for name, value in values.items():
                    scores.setdefault(name, {})[path.name] = value
            means = {}
            for name, table in scores.items():
                valid = [v for v in table.values() if v is not None]
                means[name] = sum(valid) / len(valid) if valid else None
            print(method, means)
            averages[method.name], per_view[method.name] = means, scores
        if not averages:
            raise ValueError(f"No test render sets in {scene_dir}")
        (scene_dir / "results.json").write_text(json.dumps(averages, indent=2))
        (scene_dir / "per_view.json").write_text(json.dumps(per_view, indent=2))


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
