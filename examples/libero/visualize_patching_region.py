"""Visualize localized image-token patch regions for LIBERO patching runs."""

import dataclasses
import json
import pathlib
from typing import Optional

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import tyro

from examples.libero.main_patching_expt_per_step_donor import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _patch_torch_load_for_libero_init_states,
    _preprocess_img,
)


IMAGE_BLOCKS = (
    ("base_0_rgb", "agentview", 0, 195),
    ("left_wrist_0_rgb", "wrist", 196, 391),
    ("right_wrist_0_rgb", "masked/padded", 392, 587),
)
GRID_SIZE = 14


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_goal"
    task_name_filter: Optional[str] = "put the bowl on the plate"
    patch_positions: str = "294-514"
    resize_size: int = 224
    num_steps_wait: int = 10
    seed: int = 7
    out_dir: str = "scripts_outputs_txt/patching_phase1/patched/visualizations"
    out_prefix: str = "img294-514"


def _parse_positions(spec: str) -> tuple[int, ...]:
    positions: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            positions.extend(range(int(start), int(end) + 1))
        else:
            positions.append(int(part))
    return tuple(dict.fromkeys(positions))


def _position_info(position: int) -> dict:
    for image_key, camera_name, start, end in IMAGE_BLOCKS:
        if start <= position <= end:
            local = position - start
            return {
                "position": position,
                "image_key": image_key,
                "camera": camera_name,
                "block_start": start,
                "block_end": end,
                "local_index": local,
                "row": local // GRID_SIZE,
                "col": local % GRID_SIZE,
            }
    raise ValueError(f"Position {position} is not an image-token position")


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def _representative_images(args: Args) -> dict[str, np.ndarray]:
    _patch_torch_load_for_libero_init_states()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        if args.task_name_filter and args.task_name_filter.lower() not in task.language.lower():
            continue

        initial_states = task_suite.get_task_init_states(task_id)
        env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            env.reset()
            obs = env.set_init_state(initial_states[0])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            base = _preprocess_img(obs["agentview_image"], args.resize_size)
            wrist = _preprocess_img(obs["robot0_eye_in_hand_image"], args.resize_size)
            return {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            }
        finally:
            if hasattr(env, "close"):
                env.close()

    raise ValueError(f"No task matched filter: {args.task_name_filter!r}")


def _highlight_cells(ax, infos: list[dict], image_size: int) -> None:
    cell = image_size / GRID_SIZE
    for i in range(1, GRID_SIZE):
        ax.axhline(i * cell, color="white", linewidth=0.35, alpha=0.35)
        ax.axvline(i * cell, color="white", linewidth=0.35, alpha=0.35)
    for info in infos:
        rect = patches.Rectangle(
            (info["col"] * cell, info["row"] * cell),
            cell,
            cell,
            linewidth=0.8,
            edgecolor="#ff2d00",
            facecolor="#ffb000",
            alpha=0.42,
        )
        ax.add_patch(rect)


def _summarize(infos: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for image_key, camera_name, start, end in IMAGE_BLOCKS:
        camera_infos = [info for info in infos if info["image_key"] == image_key]
        if not camera_infos:
            summary[image_key] = {
                "camera": camera_name,
                "absolute_range": [start, end],
                "selected_count": 0,
                "selected_local_ranges": [],
            }
            continue
        locals_ = sorted(info["local_index"] for info in camera_infos)
        ranges = []
        range_start = previous = locals_[0]
        for local in locals_[1:]:
            if local == previous + 1:
                previous = local
                continue
            ranges.append([range_start, previous])
            range_start = previous = local
        ranges.append([range_start, previous])
        summary[image_key] = {
            "camera": camera_name,
            "absolute_range": [start, end],
            "selected_count": len(camera_infos),
            "selected_local_ranges": ranges,
            "selected_grid_bounds": [
                {
                    "local_range": local_range,
                    "start_row_col": [local_range[0] // GRID_SIZE, local_range[0] % GRID_SIZE],
                    "end_row_col": [local_range[1] // GRID_SIZE, local_range[1] % GRID_SIZE],
                }
                for local_range in ranges
            ],
        }
    return summary


def main(args: Args) -> None:
    positions = _parse_positions(args.patch_positions)
    infos = [_position_info(position) for position in positions]
    images = _representative_images(args)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (image_key, camera_name, start, end) in zip(axes, IMAGE_BLOCKS):
        ax.imshow(images[image_key])
        camera_infos = [info for info in infos if info["image_key"] == image_key]
        _highlight_cells(ax, camera_infos, images[image_key].shape[0])
        ax.set_title(f"{image_key}\n{camera_name} tokens {start}-{end}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Localized image-token patch region: {args.patch_positions}", fontsize=13)
    png_path = out_dir / f"{args.out_prefix}_token_region_overlay.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    mapping = {
        "patch_positions": args.patch_positions,
        "grid_size": [GRID_SIZE, GRID_SIZE],
        "image_resolution": [args.resize_size, args.resize_size],
        "summary": _summarize(infos),
        "positions": infos,
        "overlay_png": str(png_path),
    }
    json_path = out_dir / f"{args.out_prefix}_token_region_mapping.json"
    json_path.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    tyro.cli(main)
