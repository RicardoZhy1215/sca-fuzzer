import sys
import os

local_path = "/home/hz25d/sca-fuzzer"
if local_path in sys.path:
    sys.path.remove(local_path)
sys.path.insert(0, local_path)

sys.path.append("/home/hz25d/sca-fuzzer/")

from SpecEnv import SpecEnv
from rvzr.tc_components.instruction import Operand,RegisterOp, MemoryOp, ImmediateOp, FlagsOp, LabelOp
from rvzr.tc_components.instruction import Instruction
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
parser.add_argument(
    "--corridor-length",
    type=int,
    default=10,
    help="The length of the corridor in fields. Note that this number includes the "
    "starting- and goal states.",
)

regs = ["RAX", "RBX", "RCX", "RDX"]
lines = [".line_1", ".line_2", ".line_3", ".line_4", ".line_5", ".line_6", ".line_7", ".line_8", ".line_9", ".line_10"]
instruction_space = []
# MOV instructions (simple, safe)
for dst in regs:
    for src in regs:
        if dst != src:   # avoid redundant MOV RAX, RAX
            MOV = Instruction("MOV", False, "", False, _section_id=10) \
                .add_op(RegisterOp(dst, 64, True, True)) \
                .add_op(RegisterOp(src, 64, True, True))
            instruction_space.append(MOV)

# XOR reg, reg
for dst in regs:
    for src in regs:
        XOR = Instruction("XOR", False, "", False, _section_id=11) \
            .add_op(RegisterOp(dst, 64, True, True)) \
            .add_op(RegisterOp(src, 64, True, True))
        instruction_space.append(XOR)

# ADD reg, 8-bit immediate
for dst in regs:
    ADD = Instruction("ADD", False, "", False, _section_id=12) \
        .add_op(RegisterOp(dst, 64, True, True)) \
        .add_op(ImmediateOp("5", 8))
    instruction_space.append(ADD)

# CMP reg, reg
for a in regs:
    for b in regs:
        CMP = Instruction("CMP", False, "", False, _section_id=13) \
            .add_op(RegisterOp(a, 64, True, True)) \
            .add_op(RegisterOp(b, 64, True, True))
        instruction_space.append(CMP)

# MOV reg, [R14]
# FIXED VERSION: MOV reg, qword ptr [R14]
for dst in regs:
    MOV_MEM = Instruction("MOV", False, "", False, _section_id=14) \
        .add_op(RegisterOp(dst, 64, True, True)) \
        .add_op(MemoryOp("R14", 64, True, False))
    instruction_space.append(MOV_MEM)



for reg in regs:
    SBB = Instruction("SBB", False, "",  False, _section_id=0) \
        .add_op(MemoryOp(reg, 64, True, True)) \
        .add_op(ImmediateOp("35", 8))
    instruction_space.append(SBB)

for op_a in regs:
    for op_b in regs:
        IMUL = Instruction("IMUL", False, "",  False, _section_id=3) \
            .add_op(MemoryOp(op_a, 8, True, False)) \
            .add_op(RegisterOp(op_b, 64, True, True), implicit=True)
        instruction_space.append(IMUL)

for line in lines:
    JNS = Instruction("JNS", False, "", True, _section_id=1) \
        .add_op(LabelOp(line))
    JMP = Instruction("JMP", False, "", True, _section_id=2) \
        .add_op(LabelOp(line))
    instruction_space.append(JNS)
    instruction_space.append(JMP)

for instr in instruction_space: print(instr)

SBB_RBX = Instruction("SBB", False, "",  False, _section_id=0) \
    .add_op(MemoryOp("RBX", 64, True, True)) \
    .add_op(ImmediateOp("35", 8))
IMUL_RCX = Instruction("IMUL", False, "",  False, _section_id=3) \
    .add_op(MemoryOp("RCX", 8, True, False)) \
    .add_op(RegisterOp("RAX", 64, True, True), implicit=True)
JNS_4 = Instruction("JNS", False, "", True, _section_id=1) \
    .add_op(LabelOp(".line_4"))
JMP_5 = Instruction("JMP", False, "", True, _section_id=2) \
    .add_op(LabelOp(".line_5"))

test_instruction_space = [SBB_RBX, JNS_4, JMP_5, IMUL_RCX]


env_config = {"instruction_space": instruction_space,
              "sequence_size": 150,
              "num_inputs": 20}

if __name__ == "__main__":
    args = parser.parse_args()
    """
    test = SpecEnv(env_config)
    print("STEP 1: \n")
    test.step(0)
    print("STEP 2: \n")
    test.step(1)
    print("STEP 3: \n")
    test.step(2)
    print("STEP 4: \n")
    test.step(3)
    exit()
    """

    """
    test = SpecEnv(env_config)
    test.testProg()
    exit()
    """
    # Can also register the env creator function explicitly with:
    # register_env("corridor-env", lambda config: SimpleCorridor())

    # Or you can hard code certain settings into the Env's constructor (`config`).
    # register_env(
    #    "corridor-env-w-len-100",
    #    lambda config: SimpleCorridor({**config, **{"corridor_length": 100}}),
    # )

    # Or allow the RLlib user to set more c'tor options via their algo config:
    # config.environment(env_config={[c'tor arg name]: [value]})
    # register_env("corridor-env", lambda config: SimpleCorridor(config))

    # base_config = (
    #     get_trainable_cls(args.algo)
    #     .get_default_config()
    #     .environment(SpecEnv, env_config=env_config)
    #     #.rollouts(num_rollout_workers=0)  
    #     .env_runners(num_env_runners=0)
    # )

    base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment(SpecEnv, env_config=env_config)
    .env_runners(num_env_runners=0)
    #.rollouts(num_rollout_workers=0)  
)
    
    run_rllib_example_script_experiment(base_config, args) 
    exit(0)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(SpecEnv, env_config=env_config)
        .env_runners(num_env_runners=0)
    )

    algo = config.build()

    for i in range(10):
        result = algo.train()
        result.pop("config")
        pprint(result)

        if i % 5 == 0:
            checkpoint_dir = algo.save_to_path()
            print(f"Checkpoint saved in directory {checkpoint_dir}")

    #exit()




    ray.init()
    algo = ppo.PPO(env=SpecEnv, config={
        "env_config": env_config  # config to pass to env class
    })

    while True:
        print(algo.train())

