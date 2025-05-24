**Overall Goal:**
LeRobot aims to be a comprehensive PyTorch-based library for real-world and simulated robotics research and application. It focuses on lowering the barrier to entry by providing tools, datasets, and pre-trained models, with a strong emphasis on imitation learning and reinforcement learning, and deep integration with the Hugging Face Hub.

**Package Structure and Functionality:**

The repository is structured to separate concerns like configuration, common utilities, specific implementations (datasets, policies, robot interfaces), and executable scripts.

1.  **`lerobot/` (Core Library Code):**
    *   **`configs/`**: This directory is central to LeRobot's flexibility. It defines dataclasses (using `draccus`) for configuring all major components of a robotics pipeline.
        *   `default.py`: Defines basic configurations like `DatasetConfig`, `WandBConfig`, `EvalConfig`.
        *   `policies.py`: Defines `PreTrainedConfig` as a base for all policy configurations. Specific policy configs (e.g., `ACTConfig`, `DiffusionConfig`) inherit from this and register themselves, allowing users to select policies via a `.type` argument.
        *   `train.py`, `eval.py`: Define the high-level pipeline configurations for training (`TrainPipelineConfig`) and evaluation (`EvalPipelineConfig`) scripts.
        *   `parser.py`: Customizes `draccus` parsing behavior, enabling features like loading configs from Hugging Face Hub paths and handling plugin discovery for custom components.
        *   Individual config files within subdirectories (e.g., `lerobot/common/envs/configs.py`, `lerobot/common/robot_devices/robots/configs.py`) define configurations for specific environments, robots, cameras, and motors.

    *   **`common/`**: Contains reusable modules and utilities.
        *   **`datasets/`**: Manages data handling.
            *   `lerobot_dataset.py`: Defines `LeRobotDataset` and `MultiLeRobotDataset` for loading and interacting with robotics datasets. It handles data fetching (local or Hub), versioning, video decoding, image transformations, and provides a standard interface. `LeRobotDatasetMetadata` handles metadata loading and management.
            *   `factory.py`: Provides `make_dataset` to instantiate datasets based on configurations.
            *   `sampler.py`: `EpisodeAwareSampler` for sampling data while respecting episode boundaries, useful for policies that require sequential context.
            *   `image_writer.py`: Asynchronous image writing to disk, crucial for high-frequency data recording without blocking the main control loop.
            *   `video_utils.py`: Utilities for encoding and decoding video frames (using `ffmpeg` via `pyav` or `torchcodec`), and getting video/audio metadata.
            *   `compute_stats.py`: Functions to calculate and aggregate statistics (mean, std, min, max) for dataset features.
            *   `online_buffer.py`: A FIFO buffer for online training, storing data in memory-mapped files for efficiency.
            *   `transforms.py`: Image augmentation and transformation classes (e.g., `RandomSubsetApply`, `SharpnessJitter`).
            *   `push_dataset_to_hub/`: (Seems to be older or utility code, as `LeRobotDataset` itself has Hub pushing capabilities).
            *   `v2/`, `v21/`: Scripts for converting datasets between different `LeRobotDataset` format versions (e.g., `convert_dataset_v1_to_v2.py`).
            *   `backward_compatibility.py`: Handles error messaging for dataset version incompatibilities.
            *   `utils.py`: General dataset utilities (flattening/unflattening dicts, JSON/JSONL I/O, creating dataset cards, validating frames/features).

        *   **`policies/`**: Contains implementations of various learning policies. Each policy typically has:
            *   `modeling_*.py`: The core PyTorch `nn.Module` defining the policy's architecture and forward pass for both training (loss computation) and inference (action selection).
            *   `configuration_*.py`: A `PreTrainedConfig` subclass defining all hyperparameters for that specific policy.
            *   Examples: `act/` (Action Chunking Transformer), `diffusion/` (Diffusion Policy), `tdmpc/` (Temporal Difference MPC), `vqbet/` (Vector Quantized BeT), `pi0/` (PaliGemma with expert), `pi0fast/` (FAST action tokenization with PaliGemma).
            *   `factory.py`: `make_policy` function to instantiate policies from configurations.
            *   `pretrained.py`: `PreTrainedPolicy` base class with Hugging Face Hub `save_pretrained` and `from_pretrained` Pytorch-like model loading.
            *   `normalize.py`: `Normalize` and `Unnormalize` modules for input/output data scaling.
            *   `utils.py`: General policy utilities.

        *   **`robot_devices/`**: Interfaces for real-world robot hardware.
            *   `robots/`:
                *   `configs.py`: Dataclasses for configuring different robots (e.g., `ManipulatorRobotConfig`, `StretchRobotConfig`, `LeKiwiRobotConfig`, specific models like `AlohaRobotConfig`, `KochRobotConfig`, `So100RobotConfig`).
                *   `manipulator.py`, `stretch.py`, `mobile_manipulator.py`: Classes abstracting specific robot types, handling their connection, calibration, teleoperation, observation capture, and action execution.
                *   `*_calibration.py` (e.g., `dynamixel_calibration.py`, `feetech_calibration.py`): Logic for calibrating specific robot arms.
                *   `utils.py`: Factory function `make_robot_from_config` and base `Robot` protocol.
            *   `cameras/`:
                *   `configs.py`: Dataclasses for camera configurations (e.g., `OpenCVCameraConfig`, `IntelRealSenseCameraConfig`).
                *   `opencv.py`, `intelrealsense.py`: Classes for interfacing with cameras using OpenCV or RealSense SDKs.
                *   `utils.py`: Factory function `make_camera_from_config`.
            *   `motors/`:
                *   `configs.py`: Dataclasses for motor bus configurations (e.g., `DynamixelMotorsBusConfig`, `FeetechMotorsBusConfig`).
                *   `dynamixel.py`, `feetech.py`: Classes for communicating with Dynamixel or Feetech motors.
                *   `utils.py`: Factory function `make_motors_bus_from_config`.
            *   `control_configs.py`: Dataclasses for different control modes (calibrate, teleoperate, record, replay, remote_robot).
            *   `control_utils.py`: Utilities for robot control loops, keyboard listeners, logging, etc.
            *   `utils.py`: General utilities for robot devices like `busy_wait` and error classes.

        *   **`envs/`**: Wrappers and configurations for simulation environments (Gymnasium-based).
            *   `configs.py`: Dataclasses for specific environment configurations (e.g., `AlohaEnv`, `PushtEnv`, `XarmEnv`).
            *   `factory.py`: `make_env` function to create Gym vector environments.
            *   `utils.py`: Utilities for preprocessing observations and mapping environment features to policy features.

        *   **`optim/`**: Optimizer and learning rate scheduler utilities.
            *   `optimizers.py`: Configuration classes for optimizers (e.g., `AdamConfig`, `AdamWConfig`) and functions to save/load optimizer states.
            *   `schedulers.py`: Configuration classes for LR schedulers (e.g., `DiffuserSchedulerConfig`, `VQBeTSchedulerConfig`) and functions to save/load scheduler states.
            *   `factory.py`: `make_optimizer_and_scheduler` to create instances from configs.

        *   **`utils/`**: General-purpose utilities used across the library.
            *   `utils.py`: Basic helpers (logging initialization, device selection, string formatting, timestamping, text-to-speech).
            *   `wandb_utils.py`: `WandBLogger` class for integrating with Weights & Biases.
            *   `hub.py`: `HubMixin` for `save_pretrained` and `push_to_hub` functionality, used by configs and policies.
            *   `io_utils.py`: Utilities for writing videos and serializing/deserializing JSON.
            *   `random_utils.py`: Functions for setting seeds and managing RNG states for reproducibility.
            *   `train_utils.py`: Utilities for saving/loading training checkpoints (policy, optimizer, scheduler, RNG states).
            *   `benchmark.py`: A `TimeBenchmark` utility for code profiling.
            *   `import_utils.py`: Checks for package availability.
            *   `logging_utils.py`: `AverageMeter` and `MetricsTracker` for tracking training/evaluation metrics.

    *   **`scripts/`**: Executable Python scripts that provide high-level functionalities. These are the primary entry points for users.
        *   `train.py`: Main script for training policies.
        *   `eval.py`: Script for evaluating trained policies in simulation.
        *   `control_robot.py`: Script for controlling real robots (teleoperation, data recording, policy deployment).
        *   `control_sim_robot.py`: (Marked as `NotImplementedError`) Intended for controlling simulated robots in a similar way to `control_robot.py`.
        *   `visualize_dataset.py`: Visualizes dataset episodes using `rerun.io`.
        *   `visualize_dataset_html.py`: Visualizes dataset episodes via a local Flask web server.
        *   `push_pretrained.py`: Script to push a trained policy (checkpoint) to the Hugging Face Hub.
        *   `configure_motor.py`, `find_motors_bus_port.py`: Utility scripts for hardware setup.
        *   `visualize_image_transforms.py`: Script to visualize the effect of image augmentations.
        *   `display_sys_info.py`: Script to print system information for debugging.

    *   `__init__.py`: Exposes lists of available environments, datasets, policies, etc., and the library version.
    *   `__version__.py`: Defines the library version.

2.  **`examples/`**: Contains Jupyter notebooks and Python scripts demonstrating how to use LeRobot's components (loading datasets, training policies, evaluating, etc.). This is a good starting point for users. The tutorials mentioned in the README (`10_use_so100.md`, `11_use_lekiwi.md`, `12_use_so101.md`) are also key parts of this.

3.  **`outputs/`** (Conceptual, not in the file list but referenced): The default directory where scripts like `train.py` and `eval.py` save their results (checkpoints, logs, videos).

4.  **`tests/`**: Contains unit tests and integration tests to ensure code correctness and stability.

5.  **`media/`**: Contains images, GIFs, and videos used in documentation and READMEs.

**LeRobot Dataset Structure**

LeRobot datasets follow a standardized file-based format designed for robotics data collection and training. The dataset is organized into three main components: **metadata** (`meta/` directory), **episode data** (`data/` directory), and **videos** (`videos/` directory). The metadata includes four key files: `info.json` contains dataset-wide information like robot type, total episodes/frames, feature schemas, and file path templates; `tasks.jsonl` stores task definitions with indices and natural language descriptions (e.g., `{"task_index": 0, "task": "Grasp a lego block and put it in the bin."}`); `episodes.jsonl` records per-episode metadata including episode indices, associated tasks, and lengths; and `episodes_stats.jsonl` contains comprehensive statistics (min, max, mean, std) for all features in each episode. The actual data is stored in **chunked parquet files** under `data/chunk-XXX/episode_XXXXXX.parquet`, where each episode's observations, actions, timestamps, and indices are stored as structured tabular data. Visual data is stored separately as **MP4 videos** organized by camera streams (e.g., `videos/chunk-000/observation.images.head/episode_000000.mp4`), with video paths templated in the info.json using format strings like `"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"`. This structure enables efficient loading of specific episodes, supports both local storage and Hugging Face Hub distribution, and maintains synchronization between tabular data and video frames through timestamp matching at the configured FPS (typically 30Hz).

**Key Design Principles:**

*   **Configuration-Driven:** Most functionalities are driven by dataclass-based configurations, allowing easy modification via CLI or config files.
*   **Modularity:** Components like datasets, policies, and robot interfaces are designed to be relatively independent and extensible.
*   **Hugging Face Hub Integration:** Datasets and models are designed to be easily shared and downloaded from the Hub.
*   **Real-World and Simulation:** Supports both real robot control and simulated environments.
*   **Standardization:** Aims to provide a standard format (`LeRobotDataset`) for robotics data.
