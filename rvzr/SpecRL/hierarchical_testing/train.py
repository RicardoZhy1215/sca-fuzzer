"""
Train SpecRL with hierarchical tuple action: (opcode, reg_src, reg_dst, imm).
Run from SpecRL: python hierarchical_testing/train.py
Or from hierarchical_testing: python train.py

Wandb: set WANDB_MODE=offline for offline, or use --specrl-wandb-project/--specrl-wandb-run
Debug: --specrl-debug-result to dump result structure on first train() iteration
"""
import sys
import os
import ctypes

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


# [V4-only] SSB / Speculative Store Bypass prctl plumbing — needed only for
# Spectre v4. Spectre v1 doesn't depend on per-process SSBD state, so the
# constants, _read_ssb_status_line, and ensure_ssb_vulnerable_for_process are
# all commented out below. Re-enable them when training v4.
# PR_GET_SPECULATION_CTRL = 52
# PR_SET_SPECULATION_CTRL = 53
# PR_SPEC_STORE_BYPASS = 0
# PR_SPEC_ENABLE = 1 << 1
#
#
# def _read_ssb_status_line() -> str:
#     """Read kernel-reported SSB status for current process."""
#     try:
#         with open("/proc/self/status", "r", encoding="utf-8") as f:
#             for line in f:
#                 if line.startswith("Speculation_Store_Bypass:"):
#                     return line.strip()
#     except OSError:
#         return "Speculation_Store_Bypass: unknown (read failed)"
#     return "Speculation_Store_Bypass: unknown (missing)"
#
#
# def ensure_ssb_vulnerable_for_process() -> None:
#     """
#     Ensure this training process allows speculative store bypass.
#     If kernel policy allows PR_SET_SPECULATION_CTRL, this sets thread to vulnerable.
#     """
#     libc = ctypes.CDLL(None, use_errno=True)
#     before = _read_ssb_status_line()
#     print(f"[SSB] Before prctl: {before}")
#
#     rc = libc.prctl(
#         ctypes.c_int(PR_SET_SPECULATION_CTRL),
#         ctypes.c_ulong(PR_SPEC_STORE_BYPASS),
#         ctypes.c_ulong(PR_SPEC_ENABLE),
#         ctypes.c_ulong(0),
#         ctypes.c_ulong(0),
#     )
#     if rc != 0:
#         err = ctypes.get_errno()
#         print(f"[SSB] prctl(PR_SPEC_ENABLE) failed (errno={err}).")
#     after = _read_ssb_status_line()
#     print(f"[SSB] After prctl: {after}")

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
    """Add end_game_count, step counters, pattern-match counts, and per-episode
    micro-architectural signal roll-ups (observable / trace divergence / leak)
    to custom_metrics so they show up in wandb under train/*."""

    @staticmethod
    def _episode_pattern_counts(env) -> dict:
        """
        Per-episode pattern match max counts (from SpecEnv).
        If on_episode_end runs *before* reset, use env.episode_pattern_match_max.
        If it runs *after* reset, that dict was cleared — use _last_completed_episode_pattern_match.
        """
        if env is None:
            return {}
        running = getattr(env, "episode_pattern_match_max", None) or {}
        if running:
            return dict(running)
        snap = getattr(env, "_last_completed_episode_pattern_match", None) or {}
        return dict(snap)

    @staticmethod
    def _episode_uarch_signals(env) -> dict:
        """
        Per-episode microarchitectural signal roll-ups. Reads whichever of
        the (running / last-completed) pairs has non-zero data, matching the
        pattern_match_counts strategy so we're order-independent w.r.t. reset.
        """
        if env is None:
            return {}
        # Prefer running values if the episode appears to have any activity.
        running_has = (
            getattr(env, "episode_trace_div_max", 0.0)
            or getattr(env, "episode_observable_any", False)
            or getattr(env, "episode_steps_observable", 0)
            or getattr(env, "episode_leak_any", False)
        )
        if running_has:
            return {
                "trace_div_score_max": float(getattr(env, "episode_trace_div_max", 0.0)),
                "observable_any": float(getattr(env, "episode_observable_any", False)),
                "steps_observable": float(getattr(env, "episode_steps_observable", 0)),
                "leak_any": float(getattr(env, "episode_leak_any", False)),
            }
        return {
            "trace_div_score_max": float(getattr(env, "_last_episode_trace_div_max", 0.0)),
            "observable_any": float(getattr(env, "_last_episode_observable_any", False)),
            "steps_observable": float(getattr(env, "_last_episode_steps_observable", 0)),
            "leak_any": float(getattr(env, "_last_episode_leak_any", False)),
        }

    def on_episode_end(self, *, episode, env=None, **kwargs):
        if env is not None and hasattr(env, "end_game_counter"):
            episode.custom_metrics["end_game_count"] = env.end_game_counter
        if env is not None and hasattr(env, "step_counter"):
            episode.custom_metrics["step_counter"] = env.step_counter
        if env is not None and hasattr(env, "succ_step_counter"):
            episode.custom_metrics["succ_step_counter"] = env.succ_step_counter
        # Pattern matcher breakdown → wandb via train result custom_metrics (mean over episodes / iter).
        counts = self._episode_pattern_counts(env)
        for key, val in counts.items():
            safe = str(key).replace("/", "_")
            episode.custom_metrics[f"pattern_matches/{safe}"] = float(val)
        # Microarchitectural signal roll-ups (leak/*).
        uarch = self._episode_uarch_signals(env)
        for key, val in uarch.items():
            episode.custom_metrics[f"leak/{key}"] = float(val)


env_config = {
    "sequence_size": 50,
    # P2.2: more inputs -> less htrace noise when judging observable / leak.
    "num_inputs": 200,
    # [V4-only] "vulnerability_type": "spectre_v4",
    "vulnerability_type": "spectre_v1",
    # Pattern shaping is generic (rule-level: same-reg store/load,
    # slow-store-fast-load, repeated-instr penalty). v1-specific templates
    # have not been added yet, so this scale only affects generic rules.
    "pattern_reward_scale": 5.0,
    "leak_reward": 600.0,
    # Single-sided observable bonus. +50 when fenced/unfenced diverge.
    "observable_bonus": 50.0,
    "observable_penalty": 0.0,
    "step_penalty": 0.05,
    # [V4-only tuning] v4 needed >=10 lea_rrr chain + store + load +
    # transmitter, hence min_steps_before_end=15. v1's canonical gadget is
    # shorter (cmp, jcc, mov[secret], mov[transmit] + a few setup mov_ri),
    # so a smaller floor is enough.
    # "min_steps_before_end": 15,
    "min_steps_before_end": 6,
    "early_end_penalty": 20.0,
    # Dense pre-leak signal (generic): rewards fenced/unfenced trace
    # divergence regardless of vulnerability type.
    "trace_divergence_reward_scale": 100.0,
    # similarity_reference_dir: pass an explicit path to a directory of v1
    # reference asm files to enable code-similarity reward. Leave unset (or
    # ""), and SpecEnv disables the similarity term but still uses the
    # generic pattern / divergence / leak signals.
    "similarity_reference_dir": "/home/hz25d/attack_seq_generation/side-channel-fuzzer/spectre_v4_example",
}

if __name__ == "__main__":
    args = parser.parse_args()
    if getattr(args, "config", None):
        CONF.load(args.config, args.include_dir)
    # [V4-only] ensure_ssb_vulnerable_for_process()
    local_debug = args.specrl_local_debug
    ray.init(
        local_mode=local_debug,
        include_dashboard=not local_debug,
        ignore_reinit_error=True,
    )
    register_specrl_hierarchical_model()

    train_config = {
        "lr": 5e-5,
        "train_batch_size": 128,
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
        "rollout_fragment_length": 16,
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
            # minibatch_size=4,
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
        for i in range(200):
            result = algo.train()
            learner_info = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
            env_runners = result.get("env_runners", {})
            custom_metrics = result.get("custom_metrics") or {}
            if not custom_metrics:
                custom_metrics = env_runners.get("custom_metrics") or {}
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
            pm_keys = {k: v for k, v in (custom_metrics or {}).items() if "pattern_matches" in str(k)}
            if pm_keys:
                print_res["pattern_matches_custom_metrics"] = pm_keys
            leak_keys = {
                k: v for k, v in (custom_metrics or {}).items()
                if "leak/" in str(k) or "leak_" in str(k)
            }
            if leak_keys:
                print_res["leak_signal_custom_metrics"] = leak_keys
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
                # RLlib aggregates custom_metrics with _mean / _min / _max suffix
                # per training iter. Forward both pattern_matches/* (gadget
                # structure) and leak/* (micro-architectural signal: observable,
                # trace_div_score_max, steps_observable, leak_any).
                for k, v in (custom_metrics or {}).items():
                    ks = str(k)
                    if "pattern_matches" in ks or "leak/" in ks or "leak_" in ks:
                        wandb_log[f"train/{k}"] = v
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
