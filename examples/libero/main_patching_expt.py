"""KV-cache patching experiment for π₀.₅ + LIBERO-Goal.

Runs the patched-corrupt condition: corrupt prompt is sent to the model, but
specified positions in the prefix KV cache are overwritten with values from a
clean-prompt donor cache harvested once before the rollout loop.

Reference: status_cc/patching_implementation.md

Usage (Phase 1, contrastive pair):
    uv run python examples/libero/main_patching_expt.py \
        --task_suite_name libero_goal \
        --task_name_filter "put_the_bowl_on_the_plate" \
        --clean_prompt "put the bowl on the plate" \
        --corrupt_prompt "put the bowl on the stove" \
        --patch_positions "594" \
        --checkpoint_dir /workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero

Sanity check (patch all positions — should recover clean behavior):
    ... --sanity_check
"""

import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional

import imageio
import jax
import jax.numpy as jnp
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import torch
import tqdm
import tyro

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _patch_torch_load_for_libero_init_states() -> None:
    """Allow LIBERO's trusted init-state pickles to load under PyTorch 2.6+.

    PyTorch 2.6 changed `torch.load` to default to `weights_only=True`.
    LIBERO's `get_task_init_states()` calls `torch.load(path)` on local
    dataset files that contain numpy objects, so the new default rejects them.
    The normal websocket baseline path uses the Python 3.8 client venv with
    torch 1.11 and does not need this compatibility shim.
    """
    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Checkpoint
    #################################################################################################################
    checkpoint_dir: str = "/workspace/openpi_assets/openpi-assets/checkpoints/pi05_libero"
    config_name: str = "pi05_libero"
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 25

    #################################################################################################################
    # Patching parameters
    #################################################################################################################
    clean_prompt: str = "put the bowl on the plate"   # donor source — clean task language
    corrupt_prompt: str = "put the bowl on the stove"  # sent to model each step
    patch_positions: str = "594"   # comma-separated absolute KV cache indices to overwrite
    sanity_check: bool = False     # if True, patches ALL 788 positions (full donor override)

    #################################################################################################################
    # Task filtering
    #################################################################################################################
    task_name_filter: Optional[str] = "put the bowl on the plate"  # only run matching tasks

    #################################################################################################################
    # Output
    #################################################################################################################
    video_out_path: str = "data/libero/videos"
    save_all_videos: bool = False
    seed: int = 7


def _preprocess_img(raw_img, resize_size):
    img = np.ascontiguousarray(raw_img[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))


def _make_state(obs):
    return np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))


def harvest_donor_kv_cache(policy: _policy.Policy, env_obs: dict, clean_prompt: str, resize_size: int):
    """Build donor KV cache once from initial env obs + clean prompt.

    Applies the full policy input transform pipeline so the donor observation
    goes through the same tokenization and normalization as a live rollout obs.
    """
    clean_obs_dict = {
        "observation/image": _preprocess_img(env_obs["agentview_image"], resize_size),
        "observation/wrist_image": _preprocess_img(env_obs["robot0_eye_in_hand_image"], resize_size),
        "observation/state": _make_state(env_obs),
        "prompt": clean_prompt,
    }

    # Apply the same input transforms as Policy.infer() does
    clean_inputs = policy._input_transform(jax.tree.map(lambda x: x, clean_obs_dict))
    clean_inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], clean_inputs)
    clean_observation = _model.Observation.from_dict(clean_inputs_jax)

    donor_kv_cache = policy._model.build_donor_kv_cache(clean_observation)
    logging.info("Donor KV cache harvested. K shape: %s", jax.tree.map(lambda x: x.shape, donor_kv_cache)[0])
    return donor_kv_cache


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    _patch_torch_load_for_libero_init_states()

    # Parse patch positions
    if args.sanity_check:
        patch_positions = tuple(range(788))
        logging.info("SANITY CHECK MODE: patching all 788 prefix positions")
    else:
        patch_positions = tuple(int(p.strip()) for p in args.patch_positions.split(","))
    logging.info("Patch positions: %s", patch_positions)

    # Build output directory name encoding the patch config
    pos_tag = "all" if args.sanity_check else args.patch_positions.replace(",", "-")
    out_dir = pathlib.Path(args.video_out_path) / ("sanity_posall" if args.sanity_check else f"pos{pos_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load policy in-process (no websocket server)
    logging.info("Loading policy from %s ...", args.checkpoint_dir)
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    logging.info("Policy loaded.")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        if args.task_name_filter and args.task_name_filter.lower() not in task_description.lower():
            continue

        logging.info("Task: %s", task_description)

        # Harvest donor KV cache once per task from episode-0 initial state.
        # Scene objects (plate, stove) are static — a t=0 donor is valid for the full rollout.
        env.reset()
        initial_obs = env.set_init_state(initial_states[0])
        # Step through the wait period so physics settle before snapshotting images
        for _ in range(args.num_steps_wait):
            initial_obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        donor_kv_cache = harvest_donor_kv_cache(policy, initial_obs, args.clean_prompt, args.resize_size)
        policy._sample_kwargs["donor_kv_cache"] = donor_kv_cache
        policy._sample_kwargs["patch_positions"] = patch_positions

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("Starting episode %d (patch_positions=%s) ...", task_episodes + 1, patch_positions)

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = _preprocess_img(obs["agentview_image"], args.resize_size)
                    wrist_img = _preprocess_img(obs["robot0_eye_in_hand_image"], args.resize_size)
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": _make_state(obs),
                            "prompt": args.corrupt_prompt,
                        }
                        # Direct in-process call — donor_kv_cache already in policy._sample_kwargs
                        action_chunk = policy.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if args.save_all_videos:
                video_filename = f"rollout_{task_segment}_ep{episode_idx:03d}_{suffix}.mp4"
            else:
                video_filename = f"rollout_{task_segment}_{suffix}.mp4"
            imageio.mimwrite(out_dir / video_filename, [np.asarray(x) for x in replay_images], fps=10)

            tqdm.tqdm.write(f"[ep {episode_idx+1:03d}] {suffix.upper()} | total {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%)")

        tqdm.tqdm.write(f"=== TASK DONE | success rate: {task_successes}/{task_episodes} ({task_successes/task_episodes*100:.1f}%) ===")

    tqdm.tqdm.write(f"=== FINAL | total success rate: {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%) ===")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
