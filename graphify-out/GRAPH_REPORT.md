# Graph Report - .  (2026-04-30)

## Corpus Check
- Large corpus: 1583 files · ~14,702,217 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 2294 nodes · 5591 edges · 59 communities detected
- Extraction: 52% EXTRACTED · 48% INFERRED · 0% AMBIGUOUS · INFERRED: 2659 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Embodiment Tags Policy|Embodiment Tags Policy]]
- [[_COMMUNITY_AI Worker Adapter|AI Worker Adapter]]
- [[_COMMUNITY_Data Interfaces|Data Interfaces]]
- [[_COMMUNITY_Embodiment Tests|Embodiment Tests]]
- [[_COMMUNITY_Dataset Statistics Actions|Dataset Statistics Actions]]
- [[_COMMUNITY_N1.7 Model Modules|N1.7 Model Modules]]
- [[_COMMUNITY_Real Robot GUI Tools|Real Robot GUI Tools]]
- [[_COMMUNITY_Model Pipeline Setup|Model Pipeline Setup]]
- [[_COMMUNITY_Data Shards Config|Data Shards Config]]
- [[_COMMUNITY_Stats Config Loading|Stats Config Loading]]
- [[_COMMUNITY_Data Utility Transforms|Data Utility Transforms]]
- [[_COMMUNITY_Image Processing|Image Processing]]
- [[_COMMUNITY_LeRobot Episode Loader|LeRobot Episode Loader]]
- [[_COMMUNITY_TensorRT Deployment|TensorRT Deployment]]
- [[_COMMUNITY_Training Evaluation|Training Evaluation]]
- [[_COMMUNITY_Policy Inference|Policy Inference]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_Simulation Wrappers|Simulation Wrappers]]
- [[_COMMUNITY_Simulation Environment Tags|Simulation Environment Tags]]
- [[_COMMUNITY_Model Architecture Image|Model Architecture Image]]
- [[_COMMUNITY_Stacked Robot Demo|Stacked Robot Demo]]
- [[_COMMUNITY_PyAV Memory Tests|PyAV Memory Tests]]
- [[_COMMUNITY_AgiBot Demo Image|AgiBot Demo Image]]
- [[_COMMUNITY_SO100 Open Loop Eval|SO100 Open Loop Eval]]
- [[_COMMUNITY_SimplerEnv Adapter|SimplerEnv Adapter]]
- [[_COMMUNITY_Reference Architecture Loop|Reference Architecture Loop]]
- [[_COMMUNITY_LIBERO Adapter|LIBERO Adapter]]
- [[_COMMUNITY_Banner Demo Image|Banner Demo Image]]
- [[_COMMUNITY_Unitree Demo Image|Unitree Demo Image]]
- [[_COMMUNITY_YAM Demo Image|YAM Demo Image]]
- [[_COMMUNITY_Header Branding|Header Branding]]
- [[_COMMUNITY_DROID Image Utils|DROID Image Utils]]
- [[_COMMUNITY_HF Local Patch|HF Local Patch]]
- [[_COMMUNITY_Synthetic Data Sources|Synthetic Data Sources]]
- [[_COMMUNITY_Data Collator|Data Collator]]
- [[_COMMUNITY_AI Worker Config|AI Worker Config]]
- [[_COMMUNITY_Hardware Targets|Hardware Targets]]
- [[_COMMUNITY_Torchcodec Smoke Test|Torchcodec Smoke Test]]
- [[_COMMUNITY_Finetune Config|Finetune Config]]
- [[_COMMUNITY_Cosmos Dreams|Cosmos Dreams]]
- [[_COMMUNITY_DROID Observation Check|DROID Observation Check]]
- [[_COMMUNITY_DROID Action Check|DROID Action Check]]
- [[_COMMUNITY_DROID Policy Reset|DROID Policy Reset]]
- [[_COMMUNITY_Video Backend Helpers|Video Backend Helpers]]
- [[_COMMUNITY_Policy Observation Check|Policy Observation Check]]
- [[_COMMUNITY_Policy Action Check|Policy Action Check]]
- [[_COMMUNITY_Policy Reset|Policy Reset]]
- [[_COMMUNITY_Tag Resolution|Tag Resolution]]
- [[_COMMUNITY_Tag Reverse Lookup|Tag Reverse Lookup]]
- [[_COMMUNITY_Quaternion WXYZ Pose|Quaternion WXYZ Pose]]
- [[_COMMUNITY_Quaternion XYZW Pose|Quaternion XYZW Pose]]
- [[_COMMUNITY_Euler Pose|Euler Pose]]
- [[_COMMUNITY_Rotation Vector Pose|Rotation Vector Pose]]
- [[_COMMUNITY_SixD Rotation Pose|SixD Rotation Pose]]
- [[_COMMUNITY_Pose 9D|Pose 9D]]
- [[_COMMUNITY_Pose 6D|Pose 6D]]
- [[_COMMUNITY_Inference Client Action|Inference Client Action]]
- [[_COMMUNITY_Policy Modality Config|Policy Modality Config]]
- [[_COMMUNITY_Sample Dataset|Sample Dataset]]

## God Nodes (most connected - your core abstractions)
1. `EmbodimentTag` - 214 edges
2. `ModalityConfig` - 132 edges
3. `Gr00tPolicy` - 122 edges
4. `LeRobotEpisodeLoader` - 115 edges
5. `PolicyClient` - 108 edges
6. `ActionFormat` - 88 edges
7. `BasePolicy` - 74 edges
8. `EndEffectorPose` - 73 edges
9. `Gr00tN1d7Config` - 70 edges
10. `JointPose` - 63 edges

## Surprising Connections (you probably didn't know these)
- `Resolve model weights (uses ``HF_TOKEN`` if the shared cache is empty).` --uses--> `Gr00tPolicy`  [INFERRED]
  tests/gr00t/policy/test_gr00t_policy_gpu.py → gr00t/policy/gr00t_policy.py
- `Build a synthetic observation that matches the policy's modality config.      Us` --uses--> `Gr00tPolicy`  [INFERRED]
  tests/gr00t/policy/test_gr00t_policy_gpu.py → gr00t/policy/gr00t_policy.py
- `End-to-end GPU inference through Gr00tPolicy.` --uses--> `Gr00tPolicy`  [INFERRED]
  tests/gr00t/policy/test_gr00t_policy_gpu.py → gr00t/policy/gr00t_policy.py
- `Action values should be finite and within a reasonable magnitude.` --uses--> `Gr00tPolicy`  [INFERRED]
  tests/gr00t/policy/test_gr00t_policy_gpu.py → gr00t/policy/gr00t_policy.py
- `Same input + same torch seed must produce the same output.` --uses--> `Gr00tPolicy`  [INFERRED]
  tests/gr00t/policy/test_gr00t_policy_gpu.py → gr00t/policy/gr00t_policy.py

## Hyperedges (group relationships)
- **GR00T N1.7 Architecture** — readme_gr00t_n17, readme_vlm_dit_architecture, readme_relative_eef_action_space [EXTRACTED 1.00]
- **Custom Embodiment Training Workflow** — data_preparation_lerobot_v2, data_preparation_modality_json, data_config_modality_config, finetune_new_embodiment [EXTRACTED 1.00]
- **Remote Policy Deployment Stack** — policy_api, policy_server_client, deployment_tensorrt_pipeline, real_world_deployment_workflow [INFERRED 0.86]
- **Multimodal Inputs Form Tokens** — model_architecture_image_observation, model_architecture_language_instruction, model_architecture_robot_state, model_architecture_image_tokens, model_architecture_text_tokens, model_architecture_action_tokens [EXTRACTED 0.90]
- **System 2 and System 1 Model Architecture** — model_architecture_vision_language_model, model_architecture_system_2, model_architecture_diffusion_transformer, model_architecture_system_1 [EXTRACTED 0.96]
- **Denoising Action Generation** — model_architecture_diffusion_transformer, model_architecture_denoising, model_architecture_action_tokens, model_architecture_motor_action [EXTRACTED 0.88]
- **Multimodal Robot Policy Purpose** — model_architecture_image_observation, model_architecture_language_instruction, model_architecture_robot_state, model_architecture_purpose_multimodal_robot_policy, model_architecture_motor_action [INFERRED 0.84]
- **Data Generation Components** — gr00t_reference_arch_diagram_data_generation, gr00t_reference_arch_diagram_nvidia_cosmos, gr00t_reference_arch_diagram_gr00t_dreams, gr00t_reference_arch_diagram_human_teleoperation_data, gr00t_reference_arch_diagram_gr00t_mimic [EXTRACTED 0.96]
- **Real Data Sources** — gr00t_reference_arch_diagram_real_data, gr00t_reference_arch_diagram_sample_dataset, gr00t_reference_arch_diagram_neural_trajectories, gr00t_reference_arch_diagram_collected_data_real [EXTRACTED 0.95]
- **Simulated Data Sources** — gr00t_reference_arch_diagram_simulated_data, gr00t_reference_arch_diagram_collected_data_simulated, gr00t_reference_arch_diagram_synthetic_motion_data, gr00t_reference_arch_diagram_nvidia_isaac_lab [EXTRACTED 0.95]
- **Post-Training Model Update** — gr00t_reference_arch_diagram_real_data, gr00t_reference_arch_diagram_simulated_data, gr00t_reference_arch_diagram_nvidia_isaac_gr00t_n_model, gr00t_reference_arch_diagram_post_training, gr00t_reference_arch_diagram_post_trained_gr00t_n_model [EXTRACTED 0.94]
- **Validation and Deployment Loop** — gr00t_reference_arch_diagram_post_trained_gr00t_n_model, gr00t_reference_arch_diagram_sil_testing, gr00t_reference_arch_diagram_nvidia_isaac_sim, gr00t_reference_arch_diagram_hil_testing, gr00t_reference_arch_diagram_deployment_on_robot, gr00t_reference_arch_diagram_data_generation, gr00t_reference_arch_diagram_accuracy_iteration_note [EXTRACTED 0.93]
- **Hardware Legend** — gr00t_reference_arch_diagram_hardware, gr00t_reference_arch_diagram_rtx_pro_server_b300, gr00t_reference_arch_diagram_rtx_pro_server, gr00t_reference_arch_diagram_jetson_thor [EXTRACTED 0.97]
- **AgiBot G1 Visible Tabletop Scene** — agibot_g1_gif, agibot_g1_visible_text, agibot_g1_robot, agibot_g1_dual_robot_arms, agibot_g1_parallel_jaw_grippers, agibot_g1_tabletop_workspace [EXTRACTED 0.95]
- **Tabletop Objects Used In Manipulation** — agibot_g1_bowl, agibot_g1_tray, agibot_g1_dishes, agibot_g1_utensils, agibot_g1_food_items, agibot_g1_yellow_sponge [EXTRACTED 0.86]
- **Household Tabletop Manipulation Demo** — agibot_g1_robot, agibot_g1_humanoid_embodiment, agibot_g1_tabletop_manipulation, agibot_g1_dish_handling_task, agibot_g1_household_assistance_purpose [INFERRED 0.78]
- **Bimanual Garment Folding Scene** — yam_dual_robot_arm_setup, yam_parallel_grippers, yam_red_garment, yam_tabletop_workspace, yam_garment_folding_task [EXTRACTED 0.90]
- **Robot Manipulation Demonstration Purpose** — yam_gif, yam_garment_folding_task, yam_bimanual_manipulation_demonstration [INFERRED 0.78]
- **Action Comparison Plot** — open_loop_eval_so100_trajectory_0, open_loop_eval_so100_ground_truth_action, open_loop_eval_so100_predicted_action, open_loop_eval_so100_inference_point, open_loop_eval_so100_action_0, open_loop_eval_so100_action_1, open_loop_eval_so100_action_2, open_loop_eval_so100_action_3, open_loop_eval_so100_action_4, open_loop_eval_so100_action_5 [EXTRACTED 0.96]
- **State Action Context** — open_loop_eval_so100_trajectory_0, open_loop_eval_so100_single_arm_gripper_state_action, open_loop_eval_so100_six_dimensional_action_trace [EXTRACTED 0.90]
- **Open Loop Evaluation Artifact** — open_loop_eval_so100_image, open_loop_eval_so100_open_loop_evaluation, open_loop_eval_so100_so100_robot_context [INFERRED 0.74]
- **Three Panel Robot Banner** — banner_banner_gif, banner_unitree_g1, banner_agibot_g1, banner_yam [EXTRACTED 1.00]
- **Multi Embodiment Robot Manipulation** — banner_unitree_g1, banner_agibot_g1, banner_yam, banner_robot_manipulation [EXTRACTED 0.91]
- **Project Purpose From Banner** — banner_vision_language_action_model, banner_generalized_humanoid_robot_skills, banner_robot_manipulation, banner_humanoid_robot [INFERRED 0.75]
- **Three-Panel Robot Demo Form** — stacked_demo_animated_robot_demonstration_gif, stacked_demo_three_panel_comparison_layout, stacked_demo_unitree_g1_humanoid_demo, stacked_demo_agibot_g1_tabletop_demo, stacked_demo_yam_robot_arm_demo [EXTRACTED 0.95]
- **Visible Robot Name Labels** — stacked_demo_unitree_g1_label, stacked_demo_agibot_g1_label, stacked_demo_yam_label, stacked_demo_unitree_g1_humanoid_demo, stacked_demo_agibot_g1_tabletop_demo, stacked_demo_yam_robot_arm_demo [EXTRACTED 0.98]
- **Robot Manipulation Demo Context** — stacked_demo_unitree_g1_humanoid_demo, stacked_demo_agibot_g1_tabletop_demo, stacked_demo_yam_robot_arm_demo, stacked_demo_pick_and_place_manipulation, stacked_demo_cloth_manipulation_task, stacked_demo_robot_manipulation_tasks [INFERRED 0.84]
- **Unitree G1 Pick And Place Scene** — unitree_g1_robot, unitree_g1_robot_hands, unitree_g1_box_object, unitree_g1_wheeled_cart, unitree_g1_shelving_unit, unitree_g1_mobile_manipulation_task [EXTRACTED 0.90]
- **Unitree G1 Humanoid Embodiment Components** — unitree_g1_robot, unitree_g1_humanoid_embodiment, unitree_g1_robot_hands, unitree_g1_head_and_cabling [EXTRACTED 0.88]
- **Unitree G1 Manipulation Demonstration Purpose** — unitree_g1_gif, unitree_g1_robot, unitree_g1_mobile_manipulation_task, unitree_g1_pick_and_place_purpose [INFERRED 0.82]
- **Isaac GR00T Header Logo** — header_compress_logo_header, header_compress_isaac_gr00t_visible_text, header_compress_humanoid_robot_icon, header_compress_neural_circuit_symbol, header_compress_green_black_branding [EXTRACTED 0.96]
- **Robot AI Project Purpose** — header_compress_isaac_gr00t_project, header_compress_robotics_concept, header_compress_vla_model_concept, header_compress_humanoid_robot_icon, header_compress_neural_circuit_symbol [INFERRED 0.70]

## Communities

### Community 0 - "Embodiment Tags Policy"
Cohesion: 0.02
Nodes (196): BasePolicy, EmbodimentTag, Embodiment tags supported by the GR00T N1.7 checkpoint.      Pretrain tags (bake, Verify that all EmbodimentTag enum values have matching configs., Every tag must have a projector index mapping., EMBODIMENT_TAG_TO_PROJECTOR_INDEX should not have orphan keys., MODALITY_CONFIGS should not have orphan keys without a matching EmbodimentTag., Posttrain tags that need built-in modality configs should have them. (+188 more)

### Community 1 - "AI Worker Adapter"
Cohesion: 0.02
Nodes (110): AiWorkerAdapter, AiWorkerCommandSender, AiWorkerObsCollector, concat_obs_rgb(), DummyCollector, DummySender, eval(), EvalConfig (+102 more)

### Community 2 - "Data Interfaces"
Cohesion: 0.02
Nodes (128): BaseProcessor, BaseProcessor, Get the shard at index idx., Get the dataset statistics. This is only required for dataloaders for robtics da, Process a list of messages and return a dictionary of model inputs.          Arg, Decode the action from the model output., Set normalization statistics., Get the modality configurations.          Returns:             dict[str, dict[st (+120 more)

### Community 3 - "Embodiment Tests"
Cohesion: 0.02
Nodes (166): resolve(), Verify that EmbodimentTag.resolve() works case-insensitively., Error message should separate base-model, posttrain, and finetuning tags., TestEmbodimentTagResolve, _dataset_path(), _model_path(), main(), Verify the DROID inference server starts and accepts connections. (+158 more)

### Community 4 - "Dataset Statistics Actions"
Cohesion: 0.03
Nodes (116): Generate dataset statistics.      Args:         dataset_path: Path to the datase, Calculate the dataset statistics of all columns for a list of parquet files., RelativeActionLoader, ActionFormat, ActionChunk, EndEffectorActionChunk, from_array(), JointActionChunk (+108 more)

### Community 5 - "N1.7 Model Modules"
Cohesion: 0.03
Nodes (84): ConfigMixin, get_action(), get_action_with_features(), get_backbone_cls(), Gr00tN1d7, Gr00tN1d7ActionHead, Huggingface will call model.train() at each training_step. To ensure         the, Forward pass through the action head.          Args:             backbone_output (+76 more)

### Community 6 - "Real Robot GUI Tools"
Cohesion: 0.03
Nodes (115): build_comparison_grid(), check_init_pose_reached(), compare_images(), compute_init_pose(), create_smooth_trajectory(), extract_reference_frame(), extract_reference_frames(), get_lang_instruction() (+107 more)

### Community 7 - "Model Pipeline Setup"
Cohesion: 0.03
Nodes (68): BasicPipeline, convert_tensors_to_lists(), ModelPipeline, Create appropriate dataset based on task and mode., Recursively convert tensors to lists in nested dictionaries/lists., A simple pipeline that works for diffusion and flowmatching-based models., Config, from_pretrained() (+60 more)

### Community 8 - "Data Shards Config"
Cohesion: 0.03
Nodes (50): ABC, reverse_lookup(), Verify PRETRAIN_TAGS, POSTTRAIN_TAGS, and FINETUNE_ONLY_TAGS are correct., Every enum member must be in exactly one category., No tag should appear in more than one category., Pretrain tags should match what's in the base model checkpoint., Verify that deprecated N1.6-only tags are fully removed., Verify reverse_lookup maps tag values back to enum names. (+42 more)

### Community 9 - "Stats Config Loading"
Cohesion: 0.03
Nodes (60): calculate_dataset_statistics(), calculate_stats_for_key(), check_stats_validity(), generate_rel_stats(), generate_stats(), main(), Validate episodes.jsonl format., Validate tasks.jsonl format. (+52 more)

### Community 10 - "Data Utility Transforms"
Cohesion: 0.05
Nodes (38): apply_sin_cos_encoding(), nested_dict_to_numpy(), normalize_values_meanstd(), normalize_values_minmax(), parse_modality_configs(), Min-max unnormalization from [-1, 1] range back to original range.      Args:, Normalize values using mean-std (z-score) normalization.      Args:         valu, Mean-std unnormalization (reverse z-score normalization).      Args:         nor (+30 more)

### Community 11 - "Image Processing"
Cohesion: 0.04
Nodes (25): BackgroundNoiseTransform, build_image_transformations(), build_image_transformations_albumentations(), FractionalCenterCrop, FractionalRandomCrop, LetterBoxPad, LetterBoxTransform, MaskedColorTransform (+17 more)

### Community 12 - "LeRobot Episode Loader"
Cohesion: 0.06
Nodes (35): Extract specific joint groups from data arrays based on modality metadata., Load and process parquet data for a specific episode.          Handles the compl, Load video data for all configured camera views at specified indices.          U, Load masks from npz/npy file at specified indices., Load mask data for all configured mask views at specified indices., Load complete episode data as a processed DataFrame.          Combines parquet d, DatasetCatalogEntry, Verify that torchcodec decodes non-identical frames for each dataset video. (+27 more)

### Community 13 - "TensorRT Deployment"
Cohesion: 0.08
Nodes (32): action_head_tensorrt_forward(), qwen3_backbone_full_trt_forward(), qwen3_backbone_llm_trt_forward(), qwen3_backbone_tensorrt_forward(), _qwen3_vit_and_scatter(), Replace Qwen3Backbone.forward() with ViT TRT + PyTorch LLM.      ViT is replaced, Replace Qwen3Backbone.forward() with PyTorch ViT + LLM TRT.      ViT stays in Py, Replace Qwen3Backbone.forward() with ViT TRT + LLM TRT.      Both ViT and LLM ar (+24 more)

### Community 14 - "Training Evaluation"
Cohesion: 0.06
Nodes (17): _batch_accuracy(), _BatchIterator, Gr00tTrainer, _PrefetchIterator, Compute token-level accuracy, ignoring ``-100`` label positions.      Args:, Trainer that bypasses torch dataloader and makes data collator async., Initialize the trainer.          Args:             *args: Positional arguments f, Return a iterable dataloader without skipping the data during resume, but reseed (+9 more)

### Community 15 - "Policy Inference"
Cohesion: 0.09
Nodes (17): device(), _build_modality_configs(), _build_observation(), policy(), _prepare_model_path(), Action values should be finite and within a reasonable magnitude., Same input + same torch seed must produce the same output., Different inputs must produce different outputs — model is not degenerate. (+9 more)

### Community 16 - "Project Documentation"
Cohesion: 0.08
Nodes (34): Isaac GR00T N1.7 Project Overview, Third-Party License Attributions, Quick-Start Development Commands, Early Access Contribution Policy, ActionConfig, ModalityConfig, LeRobot v2 Data Preparation, meta/modality.json (+26 more)

### Community 17 - "Simulation Wrappers"
Cohesion: 0.11
Nodes (15): aggregate(), compress_dict_list(), dict_take_last_n(), video_delta_indices: np.ndarray[int], please check `assert_delta_indices` to see, For video, the observation space will be (video_horizon,) + original shape, Get the maximum number of steps that we need to cache., Resets the environment using kwargs., action: dict: key-value pairs where the values are of shape (n_action_steps,) + (+7 more)

### Community 18 - "Simulation Environment Tags"
Cohesion: 0.14
Nodes (9): get_embodiment_tag_from_env_name(), Get the EmbodimentTag for a gym-registered environment name.      Looks up the e, Verify ENV_PREFIX_TO_EMBODIMENT_TAG covers all known benchmarks., Prefixes that share a common root must map to the same EmbodimentTag.          G, Test get_embodiment_tag_from_env_name() for all supported benchmarks., Only the first segment before '/' is used as the prefix., test_locomanip_g1(), TestEnvPrefixMapping (+1 more)

### Community 19 - "Model Architecture Image"
Cohesion: 0.16
Nodes (18): Action Tokens, Denoising, Model Architecture Diagram, Diffusion Transformer, Encode, Image Observation, Image Tokens, Language Instruction: "Pick up the industry object and place in yellow bin." (+10 more)

### Community 20 - "Stacked Robot Demo"
Cohesion: 0.15
Nodes (17): Visible Text: AgiBot G1, AgiBot G1 Tabletop Demo, Animated Robot Demonstration GIF, Cloth Manipulation Task, Multi-Embodiment Robotics Demo, Pick-and-Place Manipulation, Red Cloth or Garment, Robot Grippers (+9 more)

### Community 21 - "PyAV Memory Tests"
Cohesion: 0.17
Nodes (14): create_test_video(), decode_forget_close(), decode_leak_on_exception(), decode_proper_close(), decode_safe_on_exception(), get_rss_mb(), main(), Common mistake: just let the container go out of scope without closing. (+6 more)

### Community 22 - "AgiBot Demo Image"
Cohesion: 0.17
Nodes (16): Bowl, Dish Handling Task, Dishes, Dual Robot Arms, Food Items, AgiBot G1 GIF, Household Assistance Purpose, Humanoid Robot Embodiment (+8 more)

### Community 23 - "SO100 Open Loop Eval"
Cohesion: 0.22
Nodes (15): Action 0 Subplot, Action 1 Subplot, Action 2 Subplot, Action 3 Subplot, Action 4 Subplot, Action 5 Subplot, Ground Truth Action Series, Open Loop Evaluation SO100 Image (+7 more)

### Community 24 - "SimplerEnv Adapter"
Cohesion: 0.23
Nodes (2): GoogleFractalEnv, WidowXBridgeEnv

### Community 25 - "Reference Architecture Loop"
Cohesion: 0.22
Nodes (14): Note: iterate back to Data Generation if SIL, HIL, or real-world inference accuracy is not acceptable, Data Generation, Deployment on Robot, Hardware in the Loop (HIL) testing, NVIDIA Isaac GR00T N Model, NVIDIA Isaac Lab, NVIDIA Isaac Sim, Policy Deployment (+6 more)

### Community 26 - "LIBERO Adapter"
Cohesion: 0.22
Nodes (7): invert_gripper_action(), LiberoEnv, normalize_gripper_action(), quat2axisangle(), Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81, Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1]., Flips the sign of the gripper action (last dimension of action vector).     This

### Community 27 - "Banner Demo Image"
Cohesion: 0.26
Nodes (13): AgiBot G1, Banner GIF, Cloth Manipulation, Dual Arm Robot Manipulator, Generalized Humanoid Robot Skills, Humanoid Robot, Robot Manipulation, Shelf and Cart Interaction (+5 more)

### Community 28 - "Unitree Demo Image"
Cohesion: 0.21
Nodes (13): Rectangular Box Object, Unitree G1 GIF, Robot Head And External Cabling, Humanoid Robot Embodiment, Indoor Lab Environment, Mobile Manipulation Task, Pick And Place Demonstration Purpose, Unitree G1 Humanoid Robot (+5 more)

### Community 29 - "YAM Demo Image"
Cohesion: 0.32
Nodes (8): Bimanual Manipulation Demonstration, Dual Robot Arm Setup, Garment Folding Task, YAM GIF, Parallel Grippers, Red Garment, Tabletop Workspace, Visible Text: YAM

### Community 30 - "Header Branding"
Cohesion: 0.36
Nodes (8): Green on Black Branding, Humanoid Robot Icon, Isaac GR00T Project, Visible Text: Isaac GR00T, Isaac GR00T Header Logo Image, Neural Circuit Symbol, Robotics Concept, Vision-Language-Action Model Concept

### Community 31 - "DROID Image Utils"
Cohesion: 0.33
Nodes (6): convert_to_uint8(), Converts an image to uint8 if it is a float image.      This is important for re, Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a bat, Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to, resize_with_pad(), _resize_with_pad_pil()

### Community 32 - "HF Local Patch"
Cohesion: 0.4
Nodes (4): _patch_hf_local_first(), _patch_mistral(), Patch from_pretrained to prefer the local HF snapshot cache over network calls., Suppress 429 / connection errors from the HuggingFace Hub in mistral regex patch

### Community 33 - "Synthetic Data Sources"
Cohesion: 0.4
Nodes (5): Collected Data, Collected Data, GR00T Mimic, Human Teleoperation Data, Synthetic Motion Data

### Community 34 - "Data Collator"
Cohesion: 0.5
Nodes (1): BasicDataCollator

### Community 35 - "AI Worker Config"
Cohesion: 0.5
Nodes (1): # IMPORTANT: this only affects left_arm and right_arm (grippers/head/lift/base r

### Community 36 - "Hardware Targets"
Cohesion: 0.5
Nodes (4): Hardware, Jetson Thor, RTX PRO Server, RTX PRO Server / B300

### Community 37 - "Torchcodec Smoke Test"
Cohesion: 0.67
Nodes (2): Smoke test that torchcodec is importable in the CI environment., test_torchcodec_importable()

### Community 38 - "Finetune Config"
Cohesion: 0.67
Nodes (2): FinetuneConfig, Configuration for fine-tuning a Vision-Language-Action (VLA) model.      This da

### Community 39 - "Cosmos Dreams"
Cohesion: 0.67
Nodes (3): GR00T Dreams, Neural Trajectories, NVIDIA Cosmos

### Community 42 - "DROID Observation Check"
Cohesion: 1.0
Nodes (1): Check if the observation is valid.          Args:             observation: Dicti

### Community 43 - "DROID Action Check"
Cohesion: 1.0
Nodes (1): Check if the action is valid.          Args:             action: Dictionary cont

### Community 44 - "DROID Policy Reset"
Cohesion: 1.0
Nodes (1): Reset the policy to its initial state.          Args:             options: Dicti

### Community 49 - "Video Backend Helpers"
Cohesion: 1.0
Nodes (1): Return all video files under directory, sorted by path.

### Community 50 - "Policy Observation Check"
Cohesion: 1.0
Nodes (1): Check if the observation is valid.          Args:             observation: Dicti

### Community 51 - "Policy Action Check"
Cohesion: 1.0
Nodes (1): Check if the action is valid.          Args:             action: Dictionary cont

### Community 52 - "Policy Reset"
Cohesion: 1.0
Nodes (1): Reset the policy to its initial state.          Args:             options: Dicti

### Community 56 - "Tag Resolution"
Cohesion: 1.0
Nodes (1): Resolve a string to an EmbodimentTag, case-insensitively.          Matches by en

### Community 57 - "Tag Reverse Lookup"
Cohesion: 1.0
Nodes (1): Map a tag value string back to its enum name, or return the value as-is.

### Community 59 - "Quaternion WXYZ Pose"
Cohesion: 1.0
Nodes (1): Get rotation as quaternion in wxyz order (w, x, y, z)

### Community 60 - "Quaternion XYZW Pose"
Cohesion: 1.0
Nodes (1): Get rotation as quaternion in xyzw order (x, y, z, w)

### Community 61 - "Euler Pose"
Cohesion: 1.0
Nodes (1): Get rotation as Euler angles in xyz order (degrees)

### Community 62 - "Rotation Vector Pose"
Cohesion: 1.0
Nodes (1): Get rotation as rotation vector (axis-angle)

### Community 63 - "SixD Rotation Pose"
Cohesion: 1.0
Nodes (1): Get rotation as 6D representation (first two rows of rotation matrix)

### Community 64 - "Pose 9D"
Cohesion: 1.0
Nodes (1): Get pose as concatenated translation and 6D rotation (9,)

### Community 65 - "Pose 6D"
Cohesion: 1.0
Nodes (1): Get pose as concatenated translation and rotation vector (6,)

### Community 73 - "Inference Client Action"
Cohesion: 1.0
Nodes (1): Abstract method to get the action for a given state.          Args:

### Community 74 - "Policy Modality Config"
Cohesion: 1.0
Nodes (1): Return the modality config of the policy.

### Community 75 - "Sample Dataset"
Cohesion: 1.0
Nodes (1): Sample Dataset

## Ambiguous Edges - Review These
- `Food Items` → `Dish Handling Task`  [AMBIGUOUS]
  media/agibot_g1.gif · relation: conceptually_related_to
- `Yellow Sponge` → `Household Assistance Purpose`  [AMBIGUOUS]
  media/agibot_g1.gif · relation: conceptually_related_to
- `Unitree G1 GIF` → `Unclear Small Text On Box`  [AMBIGUOUS]
  media/unitree_g1.gif · relation: references

## Knowledge Gaps
- **293 isolated node(s):** `MsgSerializer`, `Recursively convert dataclasses and numpy arrays to JSON-serializable format.`, `Configuration for a modality defining how data should be sampled and loaded.`, `Set default values for action-related fields if not specified.`, `Abstract base class for robotic control policies.      This class defines the in` (+288 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `SimplerEnv Adapter`** (14 nodes): `simpler_env.py`, `GoogleFractalEnv`, `.__init__()`, `._postprocess_gripper()`, `._process_observation()`, `.reset()`, `.step()`, `register_simpler_envs()`, `WidowXBridgeEnv`, `.__init__()`, `._postprocess_gripper()`, `._process_observation()`, `.reset()`, `.step()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Collator`** (4 nodes): `BasicDataCollator`, `.__call__()`, `collators.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `AI Worker Config`** (4 nodes): `_action_rep_from_env()`, `_bool_env()`, `# IMPORTANT: this only affects left_arm and right_arm (grippers/head/lift/base r`, `ai_worker_config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Torchcodec Smoke Test`** (3 nodes): `test_torchcodec_import.py`, `Smoke test that torchcodec is importable in the CI environment.`, `test_torchcodec_importable()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Finetune Config`** (3 nodes): `FinetuneConfig`, `Configuration for fine-tuning a Vision-Language-Action (VLA) model.      This da`, `finetune_config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DROID Observation Check`** (1 nodes): `Check if the observation is valid.          Args:             observation: Dicti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DROID Action Check`** (1 nodes): `Check if the action is valid.          Args:             action: Dictionary cont`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DROID Policy Reset`** (1 nodes): `Reset the policy to its initial state.          Args:             options: Dicti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Video Backend Helpers`** (1 nodes): `Return all video files under directory, sorted by path.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Policy Observation Check`** (1 nodes): `Check if the observation is valid.          Args:             observation: Dicti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Policy Action Check`** (1 nodes): `Check if the action is valid.          Args:             action: Dictionary cont`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Policy Reset`** (1 nodes): `Reset the policy to its initial state.          Args:             options: Dicti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tag Resolution`** (1 nodes): `Resolve a string to an EmbodimentTag, case-insensitively.          Matches by en`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tag Reverse Lookup`** (1 nodes): `Map a tag value string back to its enum name, or return the value as-is.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Quaternion WXYZ Pose`** (1 nodes): `Get rotation as quaternion in wxyz order (w, x, y, z)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Quaternion XYZW Pose`** (1 nodes): `Get rotation as quaternion in xyzw order (x, y, z, w)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Euler Pose`** (1 nodes): `Get rotation as Euler angles in xyz order (degrees)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Rotation Vector Pose`** (1 nodes): `Get rotation as rotation vector (axis-angle)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SixD Rotation Pose`** (1 nodes): `Get rotation as 6D representation (first two rows of rotation matrix)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose 9D`** (1 nodes): `Get pose as concatenated translation and 6D rotation (9,)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pose 6D`** (1 nodes): `Get pose as concatenated translation and rotation vector (6,)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Inference Client Action`** (1 nodes): `Abstract method to get the action for a given state.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Policy Modality Config`** (1 nodes): `Return the modality config of the policy.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sample Dataset`** (1 nodes): `Sample Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Food Items` and `Dish Handling Task`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Yellow Sponge` and `Household Assistance Purpose`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Unitree G1 GIF` and `Unclear Small Text On Box`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `EmbodimentTag` connect `Embodiment Tags Policy` to `Data Interfaces`, `Embodiment Tests`, `Dataset Statistics Actions`, `N1.7 Model Modules`, `Real Robot GUI Tools`, `Model Pipeline Setup`, `Data Shards Config`, `Stats Config Loading`, `Data Utility Transforms`, `Simulation Environment Tags`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `ModalityConfig` connect `Data Interfaces` to `Embodiment Tags Policy`, `AI Worker Adapter`, `Embodiment Tests`, `Dataset Statistics Actions`, `Model Pipeline Setup`, `Stats Config Loading`, `Data Utility Transforms`, `LeRobot Episode Loader`, `Policy Inference`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `PolicyClient` connect `AI Worker Adapter` to `Embodiment Tags Policy`, `Data Interfaces`, `Policy Inference`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 211 inferred relationships involving `EmbodimentTag` (e.g. with `Load MODALITY_CONFIGS from embodiment_configs.py as serializable dicts.` and `Load Gr00tN1d7Config defaults.`) actually correct?**
  _`EmbodimentTag` has 211 INFERRED edges - model-reasoned connections that need verification._