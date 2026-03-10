from doctest import testfile
from rvzr.config import CONF
from sanity_check_SpecEnv import SpecEnv
from rvzr.tc_components.instruction import Instruction, RegisterOp, ImmediateOp, MemoryOp, LabelOp
from rvzr.isa_spec import InstructionSet
from rvzr import factory
from specrl_arch import register_specrl_model, build_action_to_tuple




import os
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

from model import register_custom_models, CustomMLPBackbone


# Attack instructions sequence based on revizor example
# DEC_DI  = Instruction("dec", False, "", False) \
#     .add_op(RegisterOp("rdi", 64, False, True))

MUL_RAX = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("rax", 32, False, True)) \

MUL_RDI = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("rdx", 64, False, True)) \

ADD_AL_110 = Instruction("add", False, "", False) \
    .add_op(RegisterOp("rax", 64, False, True)) \
    .add_op(ImmediateOp("-110", 64))

MOV_RBX_1 = Instruction("mov", False, "", False) \
    .add_op(MemoryOp("rbx", 64, False, True)) \
    .add_op(ImmediateOp("1", 64))


MUL_RCX = Instruction("mul", False, "", False) \
    .add_op(MemoryOp("rcx", 64, False, True)) \

test_instruction_space = [MUL_RAX, MUL_RDI, ADD_AL_110, MOV_RBX_1, MUL_RCX]

instruction_set = InstructionSet("/home/hz25d/sca-fuzzer/base.json")
unique_instruction_names = set(spec.name.lower() for spec in instruction_set.instructions)
  
opcode_vocab = list(unique_instruction_names)
reg_vocab = sorted(list(instruction_set.get_reg64_spec()))


# regs = ["rax", "rbx", "rcx", "rdx"]
# for dst in regs:
#     for src in regs:
#         if dst != src:   # avoid redundant MOV RAX, RAX
#             MOV = Instruction("mov", False, "", False) \
#                 .add_op(RegisterOp(dst, 64, True, True)) \
#                 .add_op(RegisterOp(src, 64, True, True))
#             test_instruction_space.append(MOV)

# for dst in regs:
#     for src in regs:
#         XOR = Instruction("xor", False, "", False) \
#             .add_op(RegisterOp(dst, 64, True, True)) \
#             .add_op(RegisterOp(src, 64, True, True))
#         test_instruction_space.append(XOR)


# for dst in regs:
#     ADD = Instruction("add", False, "", False) \
#         .add_op(RegisterOp(dst, 64, True, True)) \
#         .add_op(ImmediateOp("5", 8))
#     test_instruction_space.append(ADD)

# for a in regs:
#     for b in regs:
#         CMP = Instruction("cmp", False, "", False) \
#             .add_op(RegisterOp(a, 64, True, True)) \
#             .add_op(RegisterOp(b, 64, True, True))
#         test_instruction_space.append(CMP)

# for reg in regs:
#     SBB = Instruction("sbb", False, "",  False) \
#         .add_op(MemoryOp(reg, 64, True, True)) \
#         .add_op(ImmediateOp("35", 8))
#     test_instruction_space.append(SBB)

# for op_a in regs:
#     for op_b in regs:
#         IMUL = Instruction("imul", False, "",  False) \
#             .add_op(MemoryOp(op_a, 8, True, False)) \
#             .add_op(RegisterOp(op_b, 64, True, True), implicit=True)
#         test_instruction_space.append(IMUL)


env_config = {"instruction_space": test_instruction_space,
              "sequence_size": 50,
              "num_inputs": 20}



if __name__ == "__main__":
    args = parser.parse_args()
    # test = SpecEnv(env_config)
    # test.step(0)
    # test.step(1)
    # test.step(2)
    # test.step(3)


    ray.init()
    register_custom_models() 

    register_specrl_model()
    action_to_tuple = build_action_to_tuple(test_instruction_space, opcode_vocab, reg_vocab)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=SpecEnv,
            env_config=env_config
        )
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1
        )
        .training(
            lr=5e-5,
            train_batch_size=4000,
            gamma=0.99,
            model={"custom_model": "SpecRLModel", 
                        "custom_model_config": {
                        "seq_size": 50, "num_inputs": 20,
                        "action_to_tuple": action_to_tuple,
                        "hidden_dim": 256,
                        "use_dict_obs": True
                        },
                    "_disable_preprocessor_api": True,
                    },
            # model={
            #     "custom_model": "CustomMLPBackbone",
            #     "custom_model_config": {
            #         "hidden_sizes": [512, 256, 128],
            #         "activation": "relu",
            #     }
            # },
        )
        .resources(num_gpus=1)
    )
    
    algo = config.build()
    print(algo.get_policy().model)
    
    try:
        for i in range(10):
            result = algo.train()
    
            print_res = {
                "episode_reward_mean": result.get("episode_reward_mean"),
                "episodes_total": result.get("episodes_total"),
                "training_iteration": result.get("training_iteration"),
                "policy_loss": result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("policy_loss")
            }
            pprint(print_res)
    
            if i % 10 == 0:
                checkpoint_dir = algo.save()
                print(f"Iteration {i}: Checkpoint saved at {checkpoint_dir}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        algo.stop()
        ray.shutdown()





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
