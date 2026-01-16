import numpy as np
import subprocess
import gymnasium as gym
import uuid
from gymnasium import spaces
from typing import List, Optional
from rvzr.code_generator import Printer
from rvzr.arch.x86.generator import _X86Printer
from ..tc_components.test_case_code import TestCaseProgram
from ..executor import Executor
from ..model import Model
from ..analyser import Analyser
from ..data_generator import DataGenerator
from ..tc_components.test_case_data import InputData
from ..tc_components.instruction import Instruction
from rvzr.arch.x86.executor import X86IntelExecutor
from rvzr.config import CONF
from rvzr.arch.x86.target_desc import X86TargetDesc

from rvzr.arch.x86.generator import _X86NonCanonicalAddressPass,_X86PatchOpcodesPass, \
_X86SandboxPass,_X86PatchUndefinedFlagsPass,_X86PatchUndefinedResultPass,_X86U2KAccessPass


import tempfile
import os
import shutil
from subprocess import run

class SpecEnv(gym.Env):
    metadata = {}
    printer: Printer
    testcaseprogram: TestCaseProgram
    executor: Executor
    model: Model
    input_gen: DataGenerator   
    inputs: List[InputData]
    instruction_space: List[Instruction]
    misspec: bool
    observable: bool
    leak: bool
    num_steps: int
    max_steps: int
    end_game: int
    seq_size: int
    num_inputs: int
    max_trace_len: int
    bad_case: bool
    counter: int
    step_counter: int
    succ_step_counter: int

    asm_path = ""
    bin_path = ""

    def __init__(self,env_config):
        self.instruction_space = env_config['instruction_space']
        self.seq_size = env_config["sequence_size"]
        self.num_inputs = env_config["num_inputs"]
        self.bad_case = False # not needed, was for sanity check testing
        self.counter = 0 # counter for how many test cases (called @ every reset)
        self.step_counter = 0 # counter for how many steps
        self.succ_step_counter = 0 # counter for how many successful steps

        """
        ACTION SPACE:
        for now, basic action space. Simply just the subset of the ISA + an "end game" instruction 
        """
        self.action_space = spaces.Discrete(len(self.instruction_space) + 1)
        self.end_game = len(self.instruction_space)
        self.num_steps = 0
        self.max_steps = self.seq_size        

        """
        OBSERVATION SPACE:
        contains entire instruction sequence
        each element in the sequence includes:
        1. Instruction
        2. HTrace
        3. CTrace
        4. INT_MISC.RECOVERY_CYCLES (counts recovery cycles for machine clears and branch mispredictions)
        5. UOPS_ISSUED.ANY - UOPS.RETIRED.ANY (# of transient micro-operations) 
        Instruction - is an integer 
        HTrace - is a sequence of observations, or an array of integers
        CTrace - is a sequence of observations, or an array of integers
        Recovery Cycles - is an integer 
        Transient uops - is an integer 
        implemented as such - Spaces.Dict, can't do variable length so just have a max instruction sequence length 
        (HTrace, CTrace can't go past this length)
        note: -1 is a filler number for padding
        """

        min_address = np.iinfo(np.int64).min # need to figure out what's going on with htrace/ctrace addresses, as base_address and htrace addresses don't match
        max_address = np.iinfo(np.int64).max # so for now just giving (h/c)traces the set {-1, 0, 1, 2...}
        self.max_trace_len = self.seq_size * self.num_inputs
        self.observation_space = spaces.Dict(
            {
                "instruction": spaces.Box(low = -1, high = len(self.instruction_space) + 1, shape = (self.seq_size,), dtype=np.int64),
                "htrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.max_trace_len), dtype=np.int64),
                "ctrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.max_trace_len), dtype=np.int64),
                "recovery_cycles": spaces.Box(low = -1, high = np.iinfo(np.int32).max, shape = (self.seq_size, self.num_inputs), dtype=np.int64),
                "transient_uops": spaces.Box(low = -1, high = np.iinfo(np.int32).max, shape = (self.seq_size, self.num_inputs), dtype=np.int64)
            }
        )
        print(f"observation space: {self.observation_space}")

        # initialize Printer, Program, Executor, Model, Analyzer, Input Generator
        self.printer = _X86Printer() # using x86 printer for now, may need to change later
        self.program = TestCaseProgram(self.asm_path) #Initialization may need to pass in more args later compare to orignal SpecEnv
        self.executor = X86IntelExecutor()
        self.executor.valid_mem_base = 0x0
        self.executor.valid_mem_limit = 0x100000
        self.addr_mask = self.executor.valid_mem_limit - 1
        import factory
        self.model = factory.get_model(self.executor.read_base_addresses())
        print(f"\nSandbox Base Address and Code Base Address (base 10): {self.executor.read_base_addresses()}\n")
        self.analyser = factory.get_analyser()
        self.input_gen = factory.get_data_generator(CONF.input_gen_seed)
        self.inputs = self.input_gen.generate(self.num_inputs) # at some point would like these inputs to fall under action space

    """
    step(action):
    
    returns a 5 element tuple (observation, reward, terminated, truncated, info)
    how stepping works:
    action = index of instruction to append
    note that the max index is the terminate run instruction
    runs checks and instruments action as necessary, in order to sandbox it safely
    calls _get_obs and _reward to get observation and reward
    most of the logic is implemented in _get_obs, _reward, etc...
    """
    def step(self, action):
        end = (action == self.end_game)
        truncate = (self.num_steps >= self.max_steps)
        if not end and not truncate:
            inst_action = self.instruction_space[action]

            self.step_counter += 1
            print(f"NUMBER OF STEPS: {self.step_counter}")

            # run checks / instrument
            target_desc = X86TargetDesc()
            #stop workinng here
            passed_inst = X86CheckAll(self.program, inst_action, target_desc)

            passed_loop = self._infiniteLoopCheck(self.program, inst_action, 1)
            if (not passed_inst):
                print("DIDN'T PASS INSTRUCTION CHECK, NOT A VALID INSTRUCTION, THROWING AWAY")
                step_obs = self._get_obs()
                step_reward = -20
                return (step_obs, step_reward, end, truncate, {"program": self.program})
            elif (not passed_loop):
                print("DIDN'T PASS INFINITE LOOP CHECK, THROWING AWAY STEP")
                step_obs = self._get_obs()
                step_reward = -20
                return (step_obs, step_reward, end, truncate, {"program": self.program})
            else:
                self.program.append(self.instruction_space[action])
                print(f"adding step {self.instruction_space[action]}")
                self.succ_step_counter += 1
                print(f"NUMBER OF SUCCESSFUL STEPS: {self.succ_step_counter}")
        print()
        print("#=======================================================#")
        print("program: ")
        self.program.print()
        step_obs = self._get_obs()
        step_reward = self._reward()
        print(f"reward: {step_reward}")
        print("#=======================================================#")
        return (step_obs, step_reward, end, truncate, {"program": self.program})





