exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'step_timeout': None,
            'auto_reset': True,
            'reset_timeout': None,
            'retry_type': 'reset',
            'retry_waiting_time': 0.1,
            'shared_memory': False,
            'copy_on_get': True,
            'context': 'fork',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'reset_inplace': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict',
            'type': 'subprocess'
        },
        'stop_value': 1000000000,
        'n_evaluator_episode': 20,
        'env_id': 'Connect4',
        'battle_mode': 'play_with_bot_mode',
        'battle_mode_in_simulation_env': 'self_play_mode',
        'render_mode': None,
        'replay_path': None,
        'bot_action_type': 'rule',
        'agent_vs_human': False,
        'prob_random_agent': 0,
        'prob_expert_agent': 0,
        'prob_random_action_in_bot': 0.0,
        'screen_scaling': 9,
        'channel_last': False,
        'scale': False,
        'slip_prob': 0.2,
        'slip_to': 'adjacent',
        'simulation_slip_prob': None,
        'cfg_type': 'Connect4EnvDict',
        'type': 'connect4',
        'import_names': ['zoo.board_games.connect4.envs.connect4_env'],
        'collector_env_num': 8,
        'evaluator_env_num': 20
    },
    'policy': {
        'model': {
            'model_type': 'conv',
            'continuous_action_space': False,
            'observation_shape': (3, 6, 7),
            'self_supervised_learning_loss': False,
            'categorical_distribution': True,
            'image_channel': 3,
            'frame_stack_num': 1,
            'num_res_blocks': 1,
            'num_channels': 64,
            'reward_support_range': (-300.0, 301.0, 1.0),
            'value_support_range': (-300.0, 301.0, 1.0),
            'bias': True,
            'discrete_action_encoding_type': 'one_hot',
            'res_connection_in_dynamics': True,
            'norm_type': 'BN',
            'analysis_sim_norm': False,
            'analysis_dormant_ratio': False,
            'harmony_balance': False,
            'action_space_size': 7,
            'use_cuda_graphs': True
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'resume_training': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            }
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'figure_path': None,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stop_value': 1000000000,
                'n_episode': 20
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 4096,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'on_policy': False,
        'cuda': True,
        'multi_gpu': False,
        'bp_update_sync': True,
        'traj_len_inf': False,
        'use_wandb': False,
        'use_rnd_model': False,
        'sampled_algo': False,
        'gumbel_algo': False,
        'mcts_ctree': True,
        'collector_env_num': 8,
        'evaluator_env_num': 20,
        'env_type': 'board_games',
        'action_type': 'varied_action_space',
        'battle_mode': 'play_with_bot_mode',
        'monitor_extra_statistics': True,
        'game_segment_length': 21,
        'eval_offline': False,
        'calculate_dormant_ratio': False,
        'analysis_sim_norm': False,
        'analysis_dormant_ratio': False,
        'transform2string': False,
        'gray_scale': False,
        'use_augmentation': False,
        'augmentation': ['shift', 'intensity'],
        'ignore_done': False,
        'update_per_collect': 50,
        'replay_ratio': 0.25,
        'batch_size': 256,
        'optim_type': 'Adam',
        'learning_rate': 0.003,
        'target_update_freq': 100,
        'target_update_freq_for_intrinsic_reward': 1000,
        'weight_decay': 0.0001,
        'momentum': 0.9,
        'grad_clip_value': 0.5,
        'n_episode': 8,
        'num_segments': 8,
        'num_simulations': 50,
        'discount_factor': 1,
        'td_steps': 21,
        'num_unroll_steps': 5,
        'reward_loss_weight': 1,
        'value_loss_weight': 0.25,
        'policy_loss_weight': 1,
        'policy_entropy_weight': 0.005,
        'policy_label_smoothing': 0.0,
        'ssl_loss_weight': 0,
        'piecewise_decay_lr_scheduler': False,
        'threshold_training_steps_for_final_lr': 50000,
        'manual_temperature_decay': False,
        'threshold_training_steps_for_final_temperature': 100000,
        'fixed_temperature_value': 0.25,
        'use_ture_chance_label_in_chance_encoder': False,
        'reanalyze_noise': True,
        'reuse_search': False,
        'collect_with_pure_policy': False,
        'use_priority': False,
        'use_max_priority_for_new_data': False,
        'priority_prob_alpha': 0.6,
        'priority_prob_beta': 0.4,
        'root_dirichlet_alpha': 0.3,
        'root_noise_weight': 0.25,
        'random_collect_episode_num': 0,
        'eps': {
            'eps_greedy_exploration_in_collect': False,
            'type': 'linear',
            'start': 1.0,
            'end': 0.05,
            'decay': 100000
        },
        'cfg_type': 'MuZeroPolicyDict',
        'type': 'muzero',
        'import_names': ['lzero.policy.muzero'],
        'reanalyze_ratio': 0.0,
        'eval_freq': 2000,
        'replay_buffer_size': 100000,
        'device': 'cuda'
    },
    'exp_name': 'experiments/c4_slip20/runs/muzero_bot_seed1',
    'seed': 1
}
