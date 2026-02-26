from sanity_check_SpecEnv import SpecEnv
from rvzr.tc_components.instruction import Instruction, RegisterOp, ImmediateOp, MemoryOp, LabelOp

from pprint import pprint
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig

from typing import Optional

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env  # noqa

parser = add_rllib_example_script_args(
    default_reward=0.9, default_iters=50, default_timesteps=100000
)


# Attack instructions sequence based on revizor example
# DEC_DI  = Instruction("dec", False, "", False) \
#     .add_op(RegisterOp("rdi", 64, False, True))

MUL_RAX = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("rax", 64, False, True)) \

MUL_RDI = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("rdi", 64, False, True)) \

ADD_AL_110 = Instruction("add", False, "", False) \
    .add_op(RegisterOp("rax", 64, False, True)) \
    .add_op(ImmediateOp("-110", 64))

# JMP_EXIT = Instruction("jmp", True, "", False) \
#     .add_op(LabelOp(".exit_0"))

MOV_RBX_1 = Instruction("mov", False, "", False) \
    .add_op(MemoryOp("rbx", 64, False, True)) \
    .add_op(ImmediateOp("1", 64))

test_instruction_space = [MUL_RAX, MUL_RDI,ADD_AL_110, MOV_RBX_1]

env_config = {"instruction_space": test_instruction_space,
              "sequence_size": 100,
              "num_inputs": 20}

if __name__ == "__main__":
    args = parser.parse_args()
    test = SpecEnv(env_config)


    # ray.init()
    # config = (
    #     PPOConfig()
    #     .api_stack(
    #         enable_rl_module_and_learner=False,
    #         enable_env_runner_and_connector_v2=False,
    #     )
    #     .environment(
    #         env=SpecEnv,
    #         env_config=env_config
    #     )
    #     .env_runners(
    #         num_env_runners=1,
    #         num_envs_per_env_runner=1
    #     )
    #     .training(
    #         lr=5e-5,
    #         train_batch_size=4000,
    #         gamma=0.99,
    #     )
    #     .resources(num_gpus=1)
    # )
    #
    # algo = config.build()
    #
    # try:
    #     for i in range(100):
    #         result = algo.train()
    #
    #         print_res = {
    #             "episode_reward_mean": result.get("episode_reward_mean"),
    #             "episodes_total": result.get("episodes_total"),
    #             "training_iteration": result.get("training_iteration"),
    #             "policy_loss": result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("policy_loss")
    #         }
    #         pprint(print_res)
    #
    #         if i % 10 == 0:
    #             checkpoint_dir = algo.save()
    #             print(f"Iteration {i}: Checkpoint saved at {checkpoint_dir}")
    #
    # except KeyboardInterrupt:
    #     print("Training interrupted by user.")
    # finally:
    #     algo.stop()
    #     ray.shutdown()





# if __name__ == "__main__":
#     test = SpecEnv(env_config)
#     test.step(0)
#     test.step(1)
#     test.step(2)
#     test.step(3)
#     test.step(4)
#     test.step(5)
#     test.step(6)
#     test.step(7)
#     test.step(8)
#     test.step(9)
#     test.step(10)
#     test.step(11)
#     test.step(12)
#     test.step(13)

    # print("MOV RCX, 10")
    # test.step(6)
    # print("MOV RCX, R14")
    # test.step(4)
    # print("ADD RCX, 5")
    # test.step(7)
    # print("MOV RAX, [RCX]")
    # test.step(5)
    # exit()
