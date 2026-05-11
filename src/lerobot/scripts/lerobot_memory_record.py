# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On-robot inference + recording for memory-augmented VLAs.

``lerobot-memory-record`` mirrors the historical ``lerobot-record`` UX
(``--policy.path`` triggers inference; episodes saved to an ``eval_*``
dataset; leader teleop drives the follower during reset windows so the
operator can use the leader as a "home" pose), and works with
memory-augmented SmolVLA / Pi0.5 checkpoints out of the box.

Memory routing is automatic: the policy's ``select_action`` path
decodes ``observation.language_tokens`` back to text, hands it to the
``TaskEmbeddingCache`` (sentence-transformers), and feeds the result
as ``task_emb`` to the ``MLPPlusMemory`` layers via the existing
expert-forward plumbing.

Example (Pi0.5 + memory on a Trossen follower with leader-as-home):

.. code-block:: shell

    lerobot-memory-record \\
        --policy.path=outputs/vla-memory-5/checkpoints/015000/pretrained_model \\
        --policy.compile_model=false \\
        --robot.type=widowxai_follower_robot \\
        --robot.ip_address=192.168.10.4 \\
        --robot.id=follower \\
        --robot.max_relative_target=0.3 \\
        --robot.cameras="{base_0_rgb: {type: intelrealsense, serial_number_or_name: <SN>, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: intelrealsense, serial_number_or_name: <SN>, width: 640, height: 480, fps: 30}, right_wrist_0_rgb: {type: intelrealsense, serial_number_or_name: <SN>, width: 640, height: 480, fps: 30}}" \\
        --teleop.type=widowxai_leader_teleop \\
        --teleop.ip_address=192.168.10.2 \\
        --teleop.id=leader \\
        --policy.empty_cameras=1 \\
        --dataset.repo_id=joshwa71/eval_vla_memory_5_task0 \\
        --dataset.single_task="Pick up the red brick" \\
        --dataset.num_episodes=5 \\
        --dataset.episode_time_s=45 \\
        --dataset.reset_time_s=15 \\
        --dataset.fps=30 \\
        --dataset.push_to_hub=false \\
        --display_data=false
"""

import logging
import time
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, dataclass, field
from pprint import pformat

import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import FeatureType, PreTrainedConfig, parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.policies import (
    PreTrainedPolicy,
    get_policy_class,
    make_pre_post_processors,
    prepare_observation_for_inference,
)
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecordConfig:
    """Top-level config for ``lerobot-memory-record``.

    Mirrors :class:`lerobot.scripts.lerobot_record.RecordConfig` but adds
    optional ``--policy.path``: when present, the follower is driven by
    the policy during episodes and (optionally) by the leader during
    reset windows.
    """

    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Policy loaded from --policy.path. Required for inference; when None
    # the script behaves like the plain teleop-only `lerobot-record`.
    policy: PreTrainedConfig | None = None
    # Teleop is optional in policy mode (used for leader-as-home reset);
    # required when no policy is given (plain data collection).
    teleop: TeleoperatorConfig | None = None
    # Device override; resolved from policy.device when None.
    device: str | None = None
    # Rename map for observation keys (dataset/robot -> policy).
    rename_map: dict[str, str] = field(default_factory=dict)
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self):
        # Load policy from --policy.path if present.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None and self.teleop is None:
            raise ValueError(
                "Either --policy.path or --teleop.type must be set: "
                "policy mode runs inference on the follower (with optional leader teleop for reset), "
                "teleop-only mode mirrors `lerobot-record`."
            )

        # Validate dataset name prefix.
        repo_name = self.dataset.repo_id.split("/", 1)[-1]
        if self.policy is None and repo_name.startswith("eval_"):
            raise ValueError(
                f"Dataset name starts with 'eval_' ({repo_name}) but no policy is provided. "
                "Drop the prefix or pass --policy.path."
            )
        if self.policy is not None and not repo_name.startswith("eval_"):
            raise ValueError(
                f"Dataset name must start with 'eval_' when --policy.path is set (got {repo_name})."
            )

        # Propagate top-level task <-> dataset.single_task if set asymmetrically — done by
        # consumers; we keep DatasetRecordConfig as the single source of truth.

        # Resolve compute device from policy.
        if self.policy is not None:
            if self.device is None or not is_torch_device_available(self.device):
                resolved = self.policy.device
                if resolved:
                    self.device = resolved
                else:
                    self.device = auto_select_torch_device().type
                logger.info("Resolved device: %s", self.device)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _optimize_pi05_for_inference(policy: PreTrainedPolicy) -> dict:
    """Apply Pi0.5-specific inference-only memory optimizations.

    Replaces ``paligemma.lm_head`` and ``gemma_expert.lm_head`` with
    ``nn.Identity`` — both are dead weight at inference (the only references in
    ``modeling_pi05.py`` are in the load-time key rewriter). Frees ~1.6 GB.

    Applied on CPU *before* the move to the target device so the freed
    weights never touch GPU memory. Returns a dict of bytes freed for logging.

    Vision tower precision is intentionally left untouched — siglip's layer_norm
    in transformers 5.3.x rejects mixed dtypes, so partial bf16 casts break
    the forward (mixed fp32 patch embedding + bf16 layernorm). The full-FP32
    vision tower is the upstream contract for this pi05 implementation.
    """
    import torch.nn as nn

    freed = {"lm_heads": 0}
    try:
        paligemma = policy.model.paligemma_with_expert.paligemma
        gemma_expert = policy.model.paligemma_with_expert.gemma_expert
    except AttributeError:
        return freed

    for head_owner, attr in [(paligemma, "lm_head"), (gemma_expert, "lm_head")]:
        head = getattr(head_owner, attr, None)
        if isinstance(head, nn.Linear):
            freed["lm_heads"] += head.weight.numel() * head.weight.element_size()
            if head.bias is not None:
                freed["lm_heads"] += head.bias.numel() * head.bias.element_size()
            setattr(head_owner, attr, nn.Identity())

    return freed


def _load_policy(cfg: MemoryRecordConfig) -> PreTrainedPolicy:
    """Instantiate the policy from disk, honouring the saved memory config.

    ``from_pretrained`` peeks at the safetensors keys for ``mlp.mem``
    entries and calls ``attach_memory_to_*`` *before* loading weights, so
    memory-augmented SmolVLA / Pi0.5 checkpoints come back ready to run.

    For Pi0.5, the model is loaded on CPU first so we can apply
    inference-only footprint optimizations (drop unused lm_heads, bf16
    vision_tower) before moving to the target device — otherwise the
    in-init ``self.model.to(cfg.device)`` blows past available VRAM on
    smaller cards.
    """
    policy_cfg = cfg.policy
    policy_class = get_policy_class(policy_cfg.type)

    target_device = cfg.device
    is_pi05 = policy_cfg.type in {"pi0", "pi05"}

    # For Pi05 we route the build through CPU to apply inference optimizations
    # before the GPU move. For other policies (smolvla, etc.) we let the
    # original from_pretrained handle the device move directly.
    if is_pi05:
        original_device = policy_cfg.device
        policy_cfg.device = "cpu"
        try:
            policy = policy_class.from_pretrained(policy_cfg.pretrained_path, config=policy_cfg)
        finally:
            policy_cfg.device = original_device
        freed = _optimize_pi05_for_inference(policy)
        if freed.get("lm_heads"):
            logger.info(
                "Pi05 inference-mode trim: lm_heads=%.2f GB",
                freed["lm_heads"] / 1e9,
            )
    else:
        policy = policy_class.from_pretrained(policy_cfg.pretrained_path, config=policy_cfg)

    policy = policy.to(target_device)
    policy.eval()

    # Log memory-offload status when applicable.
    mem_cfg = getattr(policy_cfg, "memory_layer", None)
    offload = bool(getattr(mem_cfg, "offload_slots_to_cpu", False))
    logger.info(
        "Policy loaded: type=%s | memory_layers=%s | offload_slots_to_cpu=%s | device=%s",
        policy_cfg.type,
        bool(
            getattr(policy_cfg, "memory_layers", False)
            or getattr(mem_cfg, "enabled", False)
        ),
        offload,
        target_device,
    )
    return policy


def _resolve_action_keys(policy_cfg: PreTrainedConfig, action_keys: list[str]) -> list[str]:
    """Pick the action-key ordering used to map policy outputs back to robot motor names."""
    policy_action_names = getattr(policy_cfg, "action_feature_names", None)
    if not policy_action_names:
        return action_keys
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(action_keys):
        logger.warning(
            "policy.action_feature_names length (%d) != robot action dim (%d); using robot order",
            len(policy_action_names),
            len(action_keys),
        )
        return action_keys
    if set(policy_action_names) != set(action_keys):
        logger.warning("policy.action_feature_names keys don't match robot; using robot order")
        return action_keys
    return policy_action_names


# ---------------------------------------------------------------------------
# Per-tick inference
# ---------------------------------------------------------------------------


def _select_action(
    obs_frame: dict,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    device: torch.device,
    robot_type: str,
) -> dict[str, float]:
    """Run one full inference tick and return a robot-action dict.

    Memory routing happens inside ``policy.select_action`` via the
    existing ``task_emb`` plumbing — we don't need to thread anything
    extra here.
    """
    observation = copy(obs_frame)
    autocast_ctx = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and policy.config.use_amp
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)
        action = policy.select_action(observation)
        action = postprocessor(action)
    action_tensor = action.squeeze(0).cpu()
    action_dict = make_robot_action(action_tensor, dataset_features)
    return {k: float(action_dict[k]) for k in ordered_action_keys}


# ---------------------------------------------------------------------------
# Episode / reset loops
# ---------------------------------------------------------------------------


@safe_stop_image_writer
def _episode_loop(
    *,
    robot: Robot,
    teleop: Teleoperator | None,
    policy: PreTrainedPolicy | None,
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None,
    dataset_features: dict,
    ordered_action_keys: list[str],
    events: dict,
    fps: int,
    control_time_s: float,
    single_task: str | None,
    device: torch.device | None,
    display_data: bool,
    display_compressed_images: bool,
    use_policy: bool,
) -> None:
    """One episode of policy inference (or teleop) with optional dataset recording.

    The policy path mirrors the historical lerobot-record-with-policy
    behaviour: the follower is driven by ``select_action`` each tick, the
    leader (if attached) is held motionless and *not* read; recording is
    optional.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"Dataset fps ({dataset.fps}) != requested fps ({fps}).")

    if use_policy:
        if policy is None or preprocessor is None or postprocessor is None:
            raise ValueError("use_policy=True requires policy/preprocessor/postprocessor")
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
    elif teleop is None:
        raise ValueError("teleop-only episode requires a teleoperator")

    control_interval = 1.0 / fps
    start_episode_t = time.perf_counter()
    timestamp = 0.0
    no_action_count = 0

    while timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Always grab a fresh observation for the dataset frame + the action
        # post-processor's view of the world.
        obs_raw = robot.get_observation()
        obs_processed = robot_observation_processor(obs_raw)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Decide who produces the action this tick.
        if use_policy:
            obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            action_values = _select_action(
                obs_frame,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset_features=dataset_features,
                ordered_action_keys=ordered_action_keys,
                task=single_task or "",
                device=device,
                robot_type=robot.name,
            )
            # Run the action through the robot-side processor pipeline (clipping etc).
            robot_action_to_send = robot_action_processor((action_values, obs_raw))
        else:
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs_raw))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs_raw))
            if not action_values:
                no_action_count += 1
                if no_action_count == 1 or no_action_count % 10 == 0:
                    logger.warning("Empty teleop action — skipping tick")
                continue

        robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - loop_start
        sleep_time_s = control_interval - dt_s
        if sleep_time_s < 0:
            logger.warning(
                "Loop running slower (%.1f Hz) than target FPS (%d Hz). "
                "Common causes: (1) camera FPS, (2) policy inference, (3) CPU starvation.",
                1.0 / dt_s,
                fps,
            )

        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@safe_stop_image_writer
def _reset_loop(
    *,
    robot: Robot,
    teleop: Teleoperator | None,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    events: dict,
    fps: int,
    control_time_s: float,
    display_data: bool,
    display_compressed_images: bool,
) -> None:
    """Leader-as-home reset window: follower tracks the (still) leader.

    No dataset writes here.  If no teleop is attached the function
    becomes a wall-clock sleep — useful when the operator wants to
    physically reset the scene without the follower moving.
    """
    if teleop is None:
        precise_sleep(control_time_s)
        return

    control_interval = 1.0 / fps
    start_t = time.perf_counter()
    timestamp = 0.0
    while timestamp < control_time_s:
        loop_start = time.perf_counter()
        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs_raw = robot.get_observation()
        obs_processed = robot_observation_processor(obs_raw)

        act = teleop.get_action()
        act_processed_teleop = teleop_action_processor((act, obs_raw))
        action_values = act_processed_teleop
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs_raw))
        robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(control_interval - dt_s, 0.0))
        timestamp = time.perf_counter() - start_t


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


@parser.wrap()
def record(
    cfg: MemoryRecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logger.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="memory-record", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    use_policy = cfg.policy is not None

    # --- Hardware --------------------------------------------------------
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # --- Robot-side processors -------------------------------------------
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    # --- Feature reconciliation -----------------------------------------
    # Dataset features used both for dataset creation and for shape
    # introspection (the policy's pre-processor reads tensor shapes from
    # this dict).
    all_obs_features = robot.observation_features
    observation_features_hw = {
        k: v
        for k, v in all_obs_features.items()
        if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
    }
    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}

    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=cfg.dataset.video,
    )
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=cfg.dataset.video,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)

    raw_action_keys = list(action_features_hw.keys())
    ordered_action_keys = (
        _resolve_action_keys(cfg.policy, raw_action_keys) if use_policy else raw_action_keys
    )

    # When no rename_map is active, validate visual feature naming agreement.
    if use_policy and not cfg.rename_map:
        expected_visuals = {
            k for k, v in cfg.policy.input_features.items() if v.type == FeatureType.VISUAL
        }
        provided_visuals = {
            f"observation.images.{k}"
            for k, v in robot.observation_features.items()
            if isinstance(v, tuple)
        }
        # Allow empty_camera_* keys to be policy-provided (synthesised internally).
        policy_only_extras = {k for k in expected_visuals if "empty_camera" in k}
        expected_visuals -= policy_only_extras

        policy_subset = expected_visuals.issubset(provided_visuals)
        hw_subset = provided_visuals.issubset(expected_visuals)
        if not (policy_subset or hw_subset):
            raise ValueError(
                "Visual feature mismatch between policy and robot hardware.\n"
                f"Policy expects: {expected_visuals}\n"
                f"Robot provides: {provided_visuals}\n"
                "Either rename robot cameras to match, or pass --rename_map=..."
            )

    # --- Dataset ---------------------------------------------------------
    dataset: LeRobotDataset | None = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                    if num_cameras > 0
                    else 0
                ),
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # --- Policy + processors (after dataset so stats are available) --
        policy: PreTrainedPolicy | None = None
        preprocessor: PolicyProcessorPipeline | None = None
        postprocessor: PolicyProcessorPipeline | None = None
        device_obj: torch.device | None = None
        if use_policy:
            policy = _load_policy(cfg)
            device_obj = torch.device(cfg.device)
            dataset_stats = rename_stats(dataset.meta.stats, cfg.rename_map) if dataset else None
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=dataset_stats,
                preprocessor_overrides={
                    "device_processor": {"device": cfg.device},
                    "rename_observations_processor": {"rename_map": cfg.rename_map},
                },
            )

        # --- Connect hardware --------------------------------------------
        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logger.info(
                "Streaming encoding disabled. For faster episode saving consider "
                "--dataset.streaming_encoding=true --dataset.encoder_threads=2"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

                _episode_loop(
                    robot=robot,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    dataset_features=dataset_features,
                    ordered_action_keys=ordered_action_keys,
                    events=events,
                    fps=cfg.dataset.fps,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    device=device_obj,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    use_policy=use_policy,
                )

                # Reset window — skip after the last episode.
                if not events["stop_recording"] and (
                    recorded_episodes < cfg.dataset.num_episodes - 1 or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    _reset_loop(
                        robot=robot,
                        teleop=teleop,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        events=events,
                        fps=cfg.dataset.fps,
                        control_time_s=cfg.dataset.reset_time_s,
                        display_data=cfg.display_data,
                        display_compressed_images=display_compressed_images,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset is not None:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logger.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)

    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
