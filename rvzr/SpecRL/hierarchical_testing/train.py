"""
Train SpecRL with hierarchical tuple action: (opcode, reg_src, reg_dst, imm).
Run from SpecRL: python hierarchical_testing/train.py
Or from hierarchical_testing: python train.py

Wandb: set WANDB_MODE=offline for offline, or use --specrl-wandb-project/--specrl-wandb-run
Debug: --specrl-debug-result to dump result structure on first train() iteration
"""
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_specrl_root = os.path.dirname(_this_dir)
_rvzr_root = os.path.dirname(_specrl_root)
_repo_root = os.path.dirname(_rvzr_root)
# Ensure SpecRL in path (for check, interfaces), then put hierarchical_testing first for local model
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _specrl_root not in sys.path:
    sys.path.insert(0, _specrl_root)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)  # local model, hi_SpecEnv, inst_space override parent

from pprint import pprint
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from hi_SpecEnv import SpecEnv
from hi_model import register_specrl_hierarchical_model
from rvzr.config import CONF

parser = add_rllib_example_script_args(
    default_reward=0.9, default_iters=50, default_timesteps=100000
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    required=False,
    help="Path to the configuration file (YAML).",
)
parser.add_argument(
    "-I",
    "--include-dir",
    type=str,
    default=".",
    required=False,
    help="Path to the directory containing files included by --config.",
)
parser.add_argument("--specrl-wandb-project", type=str, default="SpecRL-hierarchical", help="Wandb project name")
parser.add_argument("--specrl-wandb-run", type=str, default=None, help="Wandb run name")
parser.add_argument(
    "--specrl-no-wandb",
    "--no-wandb",
    dest="specrl_no_wandb",
    action="store_true",
    help="Disable wandb logging",
)
parser.add_argument(
    "--specrl-local-debug",
    action="store_true",
    help="Run Ray/RLlib in local debug mode (single process, CPU-only, easier to debug).",
)


class SpecRLCallbacks(DefaultCallbacks):
    """Add end_game_count and step counters to custom_metrics for logging."""

    def on_episode_end(self, *, episode, env=None, **kwargs):
        if env is not None and hasattr(env, "end_game_counter"):
            episode.custom_metrics["end_game_count"] = env.end_game_counter
        if env is not None and hasattr(env, "step_counter"):
            episode.custom_metrics["step_counter"] = env.step_counter
        if env is not None and hasattr(env, "succ_step_counter"):
            episode.custom_metrics["succ_step_counter"] = env.succ_step_counter


env_config = {
    "sequence_size": 50,
    "num_inputs": 20,
    "vulnerability_type": "spectre_v4",
    "pattern_reward_scale": 1.0,
    "leak_reward": 600.0,
}

if __name__ == "__main__":
    args = parser.parse_args()
    if getattr(args, "config", None):
        CONF.load(args.config, args.include_dir)
    local_debug = args.specrl_local_debug
    ray.init(
        local_mode=local_debug,
        include_dashboard=not local_debug,
        ignore_reinit_error=True,
    )
    register_specrl_hierarchical_model()

    train_config = {
        "lr": 5e-5,
        "train_batch_size": 4,
        "gamma": 0.99,
        "seq_size": 50,
        "num_inputs": 20,
        "hidden_dim": 256,
    }
    if local_debug:
        # Keep local debug runs lightweight and fully synchronous.
        train_config["train_batch_size"] = 128

    model_config = {
        "custom_model": "SpecRLHierarchicalModel",
        "custom_model_config": {
            "seq_size": train_config["seq_size"],
            "num_inputs": train_config["num_inputs"],
            "hidden_dim": train_config["hidden_dim"],
            "use_dict_obs": True,
            "instruction_embed_dim": 64,
            "trace_embed_dim": 32,
            "ar_embed_dim": 64,
            "head_hidden": 128,
        },
        "custom_action_dist": "TorchHierarchicalAutoregressiveDistribution",
        "_disable_preprocessor_api": True,
    }

    env_runner_kwargs = {
        "num_env_runners": 1,
        "num_envs_per_env_runner": 1,
        "sample_timeout_s": 7200,
        "rollout_fragment_length": 4,
    }
    if local_debug:
        env_runner_kwargs.update(
            {
                "num_env_runners": 0,
                "rollout_fragment_length": 1,
            }
        )

    num_gpus = 0 if local_debug else 1

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=SpecEnv,
            env_config=env_config,
        )
        .env_runners(**env_runner_kwargs)
        .callbacks(SpecRLCallbacks)
        .training(
            lr=train_config["lr"],
            train_batch_size=train_config["train_batch_size"],
            gamma=train_config["gamma"],
            model=model_config,
            minibatch_size=4,
        )
        .resources(num_gpus=num_gpus)
    )

    if local_debug:
        print("Running in local Ray debug mode.")
        print("Overrides: local_mode=True, num_env_runners=0, rollout_fragment_length=1, num_gpus=0")

    use_wandb = not args.specrl_no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.specrl_wandb_project,
            name=args.specrl_wandb_run,
            config=train_config,
        )

    algo = config.build()
    print(algo.get_policy().model)


    try:
        for i in range(10):
            result = algo.train()
            learner_info = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
            env_runners = result.get("env_runners", {})
            custom_metrics = result.get("custom_metrics", {})
            # RLlib 2.x: episode stats under env_runners, top-level keys may be None
            ep_reward = result.get("episode_reward_mean") or env_runners.get("episode_return_mean") or env_runners.get("episode_reward_mean")
            ep_total = result.get("episodes_total") or env_runners.get("num_episodes", 0)
            print_res = {
                "episode_reward_mean": ep_reward,
                "episodes_total": ep_total,
                "training_iteration": result.get("training_iteration"),
                "policy_loss": learner_info.get("policy_loss"),
                "end_game_count": custom_metrics.get("end_game_count"),
                "num_env_steps_sampled": result.get("num_env_steps_sampled") or result.get("info", {}).get("num_env_steps_sampled"),
            }
            pprint(print_res)

            if use_wandb:
                wandb_log = {
                    "train/episode_reward_mean": ep_reward,
                    "train/episodes_total": ep_total,
                    "train/iter": result.get("training_iteration", i),
                }
                if learner_info.get("policy_loss") is not None:
                    wandb_log["train/policy_loss"] = learner_info["policy_loss"]
                if learner_info.get("vf_loss") is not None:
                    wandb_log["train/vf_loss"] = learner_info["vf_loss"]
                if learner_info.get("entropy") is not None:
                    wandb_log["train/entropy"] = learner_info["entropy"]
                if custom_metrics.get("end_game_count") is not None:
                    wandb_log["train/end_game_count"] = custom_metrics["end_game_count"]
                wandb.log(wandb_log)

            if i > 0 and i % 10 == 0:
                checkpoint_result = algo.save()
                path = getattr(checkpoint_result, "path", None) or getattr(checkpoint_result, "location", None)
                checkpoint_path = str(path) if path is not None else str(checkpoint_result)
                print(f"Iteration {i}: Checkpoint saved at {checkpoint_path}")
                if use_wandb:
                    wandb.log({"checkpoint_saved_at_iter": i})
                    wandb.run.summary["last_checkpoint_dir"] = checkpoint_path

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if use_wandb:
            wandb.finish()
        algo.stop()
        ray.shutdown()
