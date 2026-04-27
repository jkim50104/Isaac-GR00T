# Graph Report - /home/robi/projects/Isaac-GR00T  (2026-04-27)

## Corpus Check
- 107 files · ~1,059,964 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1569 nodes · 3801 edges · 51 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 1436 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Policy Rollout & Benchmarking|Policy Rollout & Benchmarking]]
- [[_COMMUNITY_AI Worker ClientServer|AI Worker Client/Server]]
- [[_COMMUNITY_Action Chunking & Processing|Action Chunking & Processing]]
- [[_COMMUNITY_Eagle VL Transformer Backbone|Eagle VL Transformer Backbone]]
- [[_COMMUNITY_Eagle3 VL Modeling|Eagle3 VL Modeling]]
- [[_COMMUNITY_Distributed Training Setup|Distributed Training Setup]]
- [[_COMMUNITY_Deployment & Engine Build|Deployment & Engine Build]]
- [[_COMMUNITY_Data Config Loading|Data Config Loading]]
- [[_COMMUNITY_Modality & Action Configuration|Modality & Action Configuration]]
- [[_COMMUNITY_LIBERO Sim Environment|LIBERO Sim Environment]]
- [[_COMMUNITY_Image Augmentation & Processing|Image Augmentation & Processing]]
- [[_COMMUNITY_Flow Matching Action Head|Flow Matching Action Head]]
- [[_COMMUNITY_Initial Pose & Eval GUI|Initial Pose & Eval GUI]]
- [[_COMMUNITY_BEHAVIOR Sim Environment|BEHAVIOR Sim Environment]]
- [[_COMMUNITY_Eagle3 VL Processor|Eagle3 VL Processor]]
- [[_COMMUNITY_Pose Math & Rotations|Pose Math & Rotations]]
- [[_COMMUNITY_Simulation Env Checks|Simulation Env Checks]]
- [[_COMMUNITY_Multistep Sim Wrapper|Multistep Sim Wrapper]]
- [[_COMMUNITY_Inference Benchmarking|Inference Benchmarking]]
- [[_COMMUNITY_Checkpoint Sync|Checkpoint Sync]]
- [[_COMMUNITY_SimplerEnv (Google Fractal)|SimplerEnv (Google Fractal)]]
- [[_COMMUNITY_Eagle3 Fast Image Processor|Eagle3 Fast Image Processor]]
- [[_COMMUNITY_LeRobot Data Pulling|LeRobot Data Pulling]]
- [[_COMMUNITY_Deployment Guide Doc|Deployment Guide Doc]]
- [[_COMMUNITY_Data Interfaces|Data Interfaces]]
- [[_COMMUNITY_Embodiment Tag Utils|Embodiment Tag Utils]]
- [[_COMMUNITY_Data Collators|Data Collators]]
- [[_COMMUNITY_Hardware Recommendations|Hardware Recommendations]]
- [[_COMMUNITY_BEHAVIOR-1K Benchmark|BEHAVIOR-1K Benchmark]]
- [[_COMMUNITY_Modules Package Init|Modules Package Init]]
- [[_COMMUNITY_Processing Mixin Doc|Processing Mixin Doc]]
- [[_COMMUNITY_Siglip2 Position Embed Doc|Siglip2 Position Embed Doc]]
- [[_COMMUNITY_Siglip2 Forward Args Doc|Siglip2 Forward Args Doc]]
- [[_COMMUNITY_Siglip2 Returns Doc|Siglip2 Returns Doc]]
- [[_COMMUNITY_Siglip2 Example Doc|Siglip2 Example Doc]]
- [[_COMMUNITY_Observation Validation Doc|Observation Validation Doc]]
- [[_COMMUNITY_Action Validation Doc|Action Validation Doc]]
- [[_COMMUNITY_Policy Reset Doc|Policy Reset Doc]]
- [[_COMMUNITY_Pose Joints Doc|Pose: Joints Doc]]
- [[_COMMUNITY_Pose 6D to Matrix Doc|Pose: 6D to Matrix Doc]]
- [[_COMMUNITY_Pose Matrix to 6D Doc|Pose: Matrix to 6D Doc]]
- [[_COMMUNITY_Pose Translation Doc|Pose: Translation Doc]]
- [[_COMMUNITY_Pose Quat WXYZ Doc|Pose: Quat WXYZ Doc]]
- [[_COMMUNITY_Pose Quat XYZW Doc|Pose: Quat XYZW Doc]]
- [[_COMMUNITY_Pose Euler XYZ Doc|Pose: Euler XYZ Doc]]
- [[_COMMUNITY_Pose Rotvec Doc|Pose: Rotvec Doc]]
- [[_COMMUNITY_Pose Rotation Matrix Doc|Pose: Rotation Matrix Doc]]
- [[_COMMUNITY_Pose 6D Rotation Doc|Pose: 6D Rotation Doc]]
- [[_COMMUNITY_Pose 9D Concat Doc|Pose: 9D Concat Doc]]
- [[_COMMUNITY_Pose 6D Concat Doc|Pose: 6D Concat Doc]]
- [[_COMMUNITY_Pose Homogeneous Doc|Pose: Homogeneous Doc]]

## God Nodes (most connected - your core abstractions)
1. `ModalityConfig` - 105 edges
2. `PolicyClient` - 95 edges
3. `EmbodimentTag` - 86 edges
4. `LeRobotEpisodeLoader` - 76 edges
5. `BasePolicy` - 72 edges
6. `Gr00tPolicy` - 64 edges
7. `EndEffectorPose` - 64 edges
8. `JointPose` - 62 edges
9. `BaseProcessor` - 56 edges
10. `VLAStepData` - 44 edges

## Surprising Connections (you probably didn't know these)
- `PolicyClient` --implements--> `BasePolicy`  [EXTRACTED]
  getting_started/policy.md → /home/robi/projects/Isaac-GR00T/gr00t/policy/policy.py
- `Cosmos-Reason backbone` --semantically_similar_to--> `Eagle-Block2A-2B-v2 tokenizer merges`  [INFERRED] [semantically similar]
  scripts/deployment/README.md → gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/merges.txt
- `ensures gr1 and gr1_unified are the same embodiment tag` --uses--> `EmbodimentTag`  [INFERRED]
  /home/robi/projects/Isaac-GR00T/gr00t/eval/sim/env_utils.py → /home/robi/projects/Isaac-GR00T/gr00t/data/embodiment_tags.py
- `Factory that creates an infinitely nestable defaultdict.` --uses--> `ModalityConfig`  [INFERRED]
  /home/robi/projects/Isaac-GR00T/gr00t/data/dataset/lerobot_episode_loader.py → /home/robi/projects/Isaac-GR00T/gr00t/data/types.py
- `Recursively turn a (nested) defaultdict into a regular dict.` --uses--> `ModalityConfig`  [INFERRED]
  /home/robi/projects/Isaac-GR00T/gr00t/data/dataset/lerobot_episode_loader.py → /home/robi/projects/Isaac-GR00T/gr00t/data/types.py

## Hyperedges (group relationships)
- **Server-client evaluation pattern across benchmarks** — policy_run_gr00t_server, policy_policyclient, simplerenv_rollout_policy, libero_benchmark, robocasa_benchmark, behavior_benchmark [EXTRACTED 0.90]
- **Modality configuration pipeline** — data_config_modalityconfig, data_config_actionconfig, data_config_modality_json, data_preparation_modality_json, data_config_register_modality_config [EXTRACTED 0.90]
- **TensorRT inference optimization chain** — deployment_export_onnx, deployment_build_tensorrt, deployment_standalone_inference, deployment_dit [EXTRACTED 0.95]

## Communities

### Community 0 - "Policy Rollout & Benchmarking"
Cohesion: 0.03
Nodes (165): BasePolicy, Get short device name for table., Compute E2E timing as sum of components (more stable than separate measurement)., Benchmark data processing separately with proper warmup.     Data processing is, Benchmark component-wise timing.     Returns dict with times for: data_processin, Print results as a markdown table using median for robustness., Set random seed for reproducibility., Recursively convert all floating point tensors to the given dtype. (+157 more)

### Community 1 - "AI Worker Client/Server"
Cohesion: 0.03
Nodes (81): ABC, AiWorkerAdapter, AiWorkerCommandSender, AiWorkerObsCollector, concat_obs_rgb(), done(), DummyCollector, DummySender (+73 more)

### Community 2 - "Action Chunking & Processing"
Cohesion: 0.04
Nodes (83): ActionChunk, EndEffectorActionChunk, JointActionChunk, num_poses(), poses(), Convert a relative action chunking to an absolute action chunking by applying, Abstract base class for robot action chunking.      This class provides common f, Interpolate the action chunking to generate intermediate poses.         Must be (+75 more)

### Community 3 - "Eagle VL Transformer Backbone"
Cohesion: 0.04
Nodes (57): ConfigMixin, AdaLayerNorm, AlternateVLDiT, BasicTransformerBlock, DiT, __init__(), Alternate Vision-Language DiT that separates image and non-image tokens     duri, SelfAttentionTransformer (+49 more)

### Community 4 - "Eagle3 VL Modeling"
Cohesion: 0.04
Nodes (57): Eagle3_VLConfig, Serializes this instance to a Python dictionary. Override the default [`~Pretrai, GenerationMixin, Eagle3_VLForConditionalGeneration, Eagle3_VLPreTrainedModel, generate(), vit_embeds: Tensor, shape [1, N, C] or [N, C]         spatial_shapes: Tensor of, apply_rope() (+49 more)

### Community 5 - "Distributed Training Setup"
Cohesion: 0.04
Nodes (32): barrier(), get_rank(), is_dist_avail_and_initialized(), run(), setup_logging(), warn_configs(), DatasetFactory, Factory class for building training datasets. Model-agnostic. (+24 more)

### Community 6 - "Deployment & Engine Build"
Cohesion: 0.06
Nodes (56): build_engine(), main(), Build TensorRT engine from ONNX model.      Args:         onnx_path: Path to ONN, convert_file(), find_videos(), is_av1(), main(), run() (+48 more)

### Community 7 - "Data Config Loading"
Cohesion: 0.08
Nodes (34): _action_rep_from_env(), _bool_env(), # IMPORTANT: this only affects left_arm and right_arm (grippers/head/lift/base r, Config, from_pretrained(), get_default_config(), Get default configuration., Complete configuration. (+26 more)

### Community 8 - "Modality & Action Configuration"
Cohesion: 0.04
Nodes (55): ActionConfig, ActionFormat, ActionRepresentation, ActionType, delta_indices design rationale, meta/modality.json, ModalityConfig, register_modality_config (+47 more)

### Community 9 - "LIBERO Sim Environment"
Cohesion: 0.07
Nodes (27): invert_gripper_action(), LiberoEnv, normalize_gripper_action(), quat2axisangle(), LIBERO environment  This file wraps the original LIBERO as a Gymnasium environme, Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81, Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1]., Flips the sign of the gripper action (last dimension of action vector).     This (+19 more)

### Community 10 - "Image Augmentation & Processing"
Cohesion: 0.06
Nodes (21): BaseProcessor, apply_with_replay(), build_image_transformations(), build_image_transformations_albumentations(), FractionalCenterCrop, FractionalRandomCrop, LetterBoxTransform, Apply albumentations transforms to multiple images with replay functionality. (+13 more)

### Community 11 - "Flow Matching Action Head"
Cohesion: 0.06
Nodes (19): ActionEncoder, Produces a sinusoidal encoding of shape (B, T, w)     given timesteps of shape (, actions:   shape (B, T, action_dim)         timesteps: shape (B,)  -- a single s, SinusoidalPositionalEncoding, SmallMLP, swish(), Trainer, _batch_accuracy() (+11 more)

### Community 12 - "Initial Pose & Eval GUI"
Cohesion: 0.1
Nodes (31): _load_init_pose_module(), Show instruction picker, choose random matching episode, populate lang., build_comparison_grid(), check_init_pose_reached(), compare_images(), compute_init_pose(), create_smooth_trajectory(), extract_reference_frame() (+23 more)

### Community 13 - "BEHAVIOR Sim Environment"
Cohesion: 0.09
Nodes (29): BEHAVIORGr00tEnv, flatten_obs_dict(), load_task_instance_for_env(), postprocess_info(), preprocess_action(), preprocess_obs(), Process the observation dictionary by recursively flattening the keys.     so ob, Preprocess the observation dictionary before passing it to the policy. (+21 more)

### Community 14 - "Eagle3 VL Processor"
Cohesion: 0.1
Nodes (28): adjust_by_factor(), Eagle3_VLProcessor, Eagle3_VLProcessorKwargs, fetch_image(), fetch_video(), from_args_and_dict(), from_pretrained(), get_video_reader_backend() (+20 more)

### Community 15 - "Pose Math & Rotations"
Cohesion: 0.12
Nodes (27): euler_xyz(), EulerOrder, homogeneous(), invert_transformation(), _matrix_to_rot6d(), num_joints(), quat_wxyz(), quat_xyzw() (+19 more)

### Community 16 - "Simulation Env Checks"
Cohesion: 0.15
Nodes (17): check_egl_installation(), check_g1_locomanipulation_environment(), check_libero_environments(), check_robocasa_environments(), check_robocasa_gr1_tabletop_tasks_environments(), check_simpler_env_environments(), check_uv_installation(), check_vulkan_installation() (+9 more)

### Community 17 - "Multistep Sim Wrapper"
Cohesion: 0.12
Nodes (15): aggregate(), compress_dict_list(), dict_take_last_n(), video_delta_indices: np.ndarray[int], please check `assert_delta_indices` to see, For video, the observation space will be (video_horizon,) + original shape, Get the maximum number of steps that we need to cache., Resets the environment using kwargs., action: dict: key-value pairs where the values are of shape (n_action_steps,) + (+7 more)

### Community 18 - "Inference Benchmarking"
Cohesion: 0.17
Nodes (15): benchmark_components(), benchmark_data_processing(), compute_e2e_from_components(), get_device_name(), main(), prepare_model_inputs(), print_markdown_table(), _rec_to_dtype() (+7 more)

### Community 19 - "Checkpoint Sync"
Cohesion: 0.22
Nodes (20): choose_items_menu(), ckpt_sort_key(), confirm(), format_row(), list_remote_ckpts(), list_remote_dirs(), list_remote_experiments(), list_remote_hparams() (+12 more)

### Community 20 - "SimplerEnv (Google Fractal)"
Cohesion: 0.23
Nodes (3): GoogleFractalEnv, register_simpler_envs(), WidowXBridgeEnv

### Community 21 - "Eagle3 Fast Image Processor"
Cohesion: 0.19
Nodes (8): BaseImageProcessorFast, DefaultFastImageProcessorKwargs, crop(), Eagle3_VLFastImageProcessorKwargs, Eagle3_VLImageProcessorFast, preprocess(), Prepare the images structure for processing.          Args:             images (, Crop the given numpy array.          Args:         img (torch.Tensor): Image to

### Community 22 - "LeRobot Data Pulling"
Cohesion: 0.56
Nodes (7): list_remote_dirs(), main(), parse_args(), _rsync_exclude_args(), rsync_pull(), run(), ssh_lines()

### Community 23 - "Deployment Guide Doc"
Cohesion: 0.25
Nodes (9): benchmark_inference.py, build_tensorrt_engine.py, Cosmos-Reason backbone, DiT (Diffusion Transformer) action head, export_onnx_n1d6.py, GR00T Deployment & Inference Guide, standalone_inference_script.py, TensorRT optimization rationale (+1 more)

### Community 24 - "Data Interfaces"
Cohesion: 0.48
Nodes (5): collator(), get_shard(), get_shard_length(), __len__(), set_statistics()

### Community 25 - "Embodiment Tag Utils"
Cohesion: 0.57
Nodes (5): get_embodiment_tag_from_env_name(), is_behavior_env(), is_gr1_env(), is_groot_locomanip_env(), ensures gr1 and gr1_unified are the same embodiment tag

### Community 26 - "Data Collators"
Cohesion: 0.33
Nodes (1): BasicDataCollator

### Community 27 - "Hardware Recommendations"
Cohesion: 0.33
Nodes (6): NVIDIA DGX B300, GR00T-Dreams, Isaac GR00T-Mimic, Isaac GR00T N Models, Jetson AGX Thor Developer Kit, NVIDIA RTX PRO Server

### Community 28 - "BEHAVIOR-1K Benchmark"
Cohesion: 0.5
Nodes (4): BEHAVIOR-1K benchmark, GR00T-N1.6-BEHAVIOR1k checkpoint, OmniGibson / Isaac Sim, BEHAVIOR_R1_PRO embodiment tag

### Community 29 - "Modules Package Init"
Cohesion: 0.67
Nodes (1): Modules package for Groot VLA models.

### Community 39 - "Processing Mixin Doc"
Cohesion: 1.0
Nodes (1): Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dicti

### Community 40 - "Siglip2 Position Embed Doc"
Cohesion: 1.0
Nodes (1): Resize positional embeddings to image-specific size and pad to a fixed size.

### Community 41 - "Siglip2 Forward Args Doc"
Cohesion: 1.0
Nodes (1): r"""         Args:             inputs_embeds (`torch.FloatTensor` of shape `(bat

### Community 42 - "Siglip2 Returns Doc"
Cohesion: 1.0
Nodes (1): r"""         Returns:

### Community 43 - "Siglip2 Example Doc"
Cohesion: 1.0
Nodes (1): r"""         Returns:          Examples:          ```python         >>> from PIL

### Community 44 - "Observation Validation Doc"
Cohesion: 1.0
Nodes (1): Check if the observation is valid.          Args:             observation: Dicti

### Community 45 - "Action Validation Doc"
Cohesion: 1.0
Nodes (1): Check if the action is valid.          Args:             action: Dictionary cont

### Community 46 - "Policy Reset Doc"
Cohesion: 1.0
Nodes (1): Reset the policy to its initial state.          Args:             options: Dicti

### Community 48 - "Pose: Joints Doc"
Cohesion: 1.0
Nodes (1): Get the number of joints.          Returns:             Number of joints in the

### Community 49 - "Pose: 6D to Matrix Doc"
Cohesion: 1.0
Nodes (1): Convert 6D rotation representation to rotation matrix.          Args:

### Community 50 - "Pose: Matrix to 6D Doc"
Cohesion: 1.0
Nodes (1): Convert rotation matrix to 6D rotation representation.          Args:

### Community 51 - "Pose: Translation Doc"
Cohesion: 1.0
Nodes (1): Get translation vector.          Returns:             Translation array - shape

### Community 52 - "Pose: Quat WXYZ Doc"
Cohesion: 1.0
Nodes (1): Get rotation as quaternion in wxyz order (w, x, y, z)

### Community 53 - "Pose: Quat XYZW Doc"
Cohesion: 1.0
Nodes (1): Get rotation as quaternion in xyzw order (x, y, z, w)

### Community 54 - "Pose: Euler XYZ Doc"
Cohesion: 1.0
Nodes (1): Get rotation as Euler angles in xyz order (degrees)

### Community 55 - "Pose: Rotvec Doc"
Cohesion: 1.0
Nodes (1): Get rotation as rotation vector (axis-angle)

### Community 56 - "Pose: Rotation Matrix Doc"
Cohesion: 1.0
Nodes (1): Get rotation as 3x3 rotation matrix

### Community 57 - "Pose: 6D Rotation Doc"
Cohesion: 1.0
Nodes (1): Get rotation as 6D representation (first two rows of rotation matrix)

### Community 58 - "Pose: 9D Concat Doc"
Cohesion: 1.0
Nodes (1): Get pose as concatenated translation and 6D rotation (9,)

### Community 59 - "Pose: 6D Concat Doc"
Cohesion: 1.0
Nodes (1): Get pose as concatenated translation and rotation vector (6,)

### Community 60 - "Pose: Homogeneous Doc"
Cohesion: 1.0
Nodes (1): Get homogeneous transformation matrix.          Returns:             Homogeneous

## Knowledge Gaps
- **215 isolated node(s):** `Run a remote command and return non-empty stripped lines. Print debug on failure`, `List immediate child directory names under a remote path (portable).`, `Parse user input like:       "1 2 5", "1,2,5", "1-3,7", "all"     Returns 0-base`, `Show numbered list and allow user to pick indices or all.     Returns selected i`, `Read language instruction from dataset's episodes.jsonl.     Returns the task st` (+210 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Data Collators`** (6 nodes): `BasicDataCollator`, `.__call__()`, `collators.py`, `__init__.py`, `collators.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Modules Package Init`** (3 nodes): `__init__.py`, `__init__.py`, `Modules package for Groot VLA models.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Processing Mixin Doc`** (1 nodes): `Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dicti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Siglip2 Position Embed Doc`** (1 nodes): `Resize positional embeddings to image-specific size and pad to a fixed size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Siglip2 Forward Args Doc`** (1 nodes): `r"""         Args:             inputs_embeds (`torch.FloatTensor` of shape `(bat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Siglip2 Returns Doc`** (1 nodes): `r"""         Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Siglip2 Example Doc`** (1 nodes): `r"""         Returns:          Examples:          ```python         >>> from PIL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Observation Validation Doc`** (1 nodes): `Check if the observation is valid.          Args:             observation: Dicti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Action Validation Doc`** (1 nodes): `Check if the action is valid.          Args:             action: Dictionary cont`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Policy Reset Doc`** (1 nodes): `Reset the policy to its initial state.          Args:             options: Dicti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Joints Doc`** (1 nodes): `Get the number of joints.          Returns:             Number of joints in the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: 6D to Matrix Doc`** (1 nodes): `Convert 6D rotation representation to rotation matrix.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Matrix to 6D Doc`** (1 nodes): `Convert rotation matrix to 6D rotation representation.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Translation Doc`** (1 nodes): `Get translation vector.          Returns:             Translation array - shape`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Quat WXYZ Doc`** (1 nodes): `Get rotation as quaternion in wxyz order (w, x, y, z)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Quat XYZW Doc`** (1 nodes): `Get rotation as quaternion in xyzw order (x, y, z, w)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Euler XYZ Doc`** (1 nodes): `Get rotation as Euler angles in xyz order (degrees)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Rotvec Doc`** (1 nodes): `Get rotation as rotation vector (axis-angle)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Rotation Matrix Doc`** (1 nodes): `Get rotation as 3x3 rotation matrix`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: 6D Rotation Doc`** (1 nodes): `Get rotation as 6D representation (first two rows of rotation matrix)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: 9D Concat Doc`** (1 nodes): `Get pose as concatenated translation and 6D rotation (9,)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: 6D Concat Doc`** (1 nodes): `Get pose as concatenated translation and rotation vector (6,)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose: Homogeneous Doc`** (1 nodes): `Get homogeneous transformation matrix.          Returns:             Homogeneous`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PolicyClient` connect `AI Worker Client/Server` to `Policy Rollout & Benchmarking`, `Deployment & Engine Build`, `LIBERO Sim Environment`, `Initial Pose & Eval GUI`, `Simulation Env Checks`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `ModalityConfig` connect `Policy Rollout & Benchmarking` to `AI Worker Client/Server`, `Action Chunking & Processing`, `Data Config Loading`, `Simulation Env Checks`, `Inference Benchmarking`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Why does `EmbodimentTag` connect `Policy Rollout & Benchmarking` to `Action Chunking & Processing`, `Eagle VL Transformer Backbone`, `Distributed Training Setup`, `Deployment & Engine Build`, `Data Config Loading`, `Image Augmentation & Processing`, `Inference Benchmarking`, `Embodiment Tag Utils`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Are the 101 inferred relationships involving `ModalityConfig` (e.g. with `# IMPORTANT: this only affects left_arm and right_arm (grippers/head/lift/base r` and `ArgsConfig`) actually correct?**
  _`ModalityConfig` has 101 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `PolicyClient` (e.g. with `ArgsConfig` and `BaseInferenceClient`) actually correct?**
  _`PolicyClient` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 83 inferred relationships involving `EmbodimentTag` (e.g. with `# IMPORTANT: this only affects left_arm and right_arm (grippers/head/lift/base r` and `DiTInputCapture`) actually correct?**
  _`EmbodimentTag` has 83 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `LeRobotEpisodeLoader` (e.g. with `DiTInputCapture` and `Helper class to capture DiT forward pass inputs during inference.`) actually correct?**
  _`LeRobotEpisodeLoader` has 60 INFERRED edges - model-reasoned connections that need verification._