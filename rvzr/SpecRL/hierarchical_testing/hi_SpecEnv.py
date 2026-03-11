import random

import numpy as np
import subprocess
import gymnasium as gym
import uuid
from gymnasium import spaces
from copy import deepcopy
from typing import Counter, List, Optional
from rvzr.code_generator import Printer
from rvzr.arch.x86.generator import _X86Printer
from rvzr.tc_components.test_case_code import TestCaseProgram
from rvzr.tc_components.test_case_data import InputData
from rvzr.executor import Executor
from rvzr.model import Model
from rvzr.analyser import Analyser
from rvzr.data_generator import DataGenerator
from rvzr.code_generator import CodeGenerator
from rvzr.code_generator import assemble
from rvzr.arch.x86.generator import X86Generator
from rvzr.asm_parser import AsmParser
from rvzr.elf_parser import ELFParser
from rvzr.arch.x86.asm_parser import X86AsmParser
from rvzr.elf_parser import ELFParser
from rvzr.isa_spec import InstructionSet
from rvzr.tc_components.test_case_data import InputData
from rvzr.tc_components.instruction import Instruction, RegisterOp, MemoryOp, ImmediateOp
from rvzr.arch.x86.executor import X86IntelExecutor
from rvzr.config import CONF
from rvzr.code_generator import assemble
from rvzr.arch.x86.target_desc import X86TargetDesc
from check import X86CheckAll
from rvzr.traces import CTrace
from rvzr.traces import HTrace
from rvzr import factory
from rvzr.fuzzer import Fuzzer, _RoundManager
from rvzr.arch.x86.fuzzer import X86Fuzzer
from rvzr.arch.x86.fuzzer import _create_fenced_test_case
import copy
import shutil
from collections import Counter


from rvzr.arch.x86.generator import _X86NonCanonicalAddressPass,_X86PatchOpcodesPass, \
_X86SandboxPass,_X86PatchUndefinedFlagsPass,_X86PatchUndefinedResultPass,_X86U2KAccessPass


import tempfile
import os
import shutil
from subprocess import run





class SpecEnv(gym.Env):
    metadata = {}
    printer: Printer
    generator: X86Generator
    new_program: TestCaseProgram
    asm_parser: AsmParser
    elf_parser: ELFParser
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

    bin_path = "my_test_case.o"
    asm_path = "my_test_case.asm"

    def __init__(self, env_config):
        self.seq_size = env_config["sequence_size"]
        self.num_inputs = env_config["num_inputs"]
        self.bad_case = False
        self.counter = 0
        self.step_counter = 0
        self.succ_step_counter = 0

        # Hierarchical mode: action_to_tuple instead of instruction_space
        if "action_to_tuple" in env_config:
            self._action_to_tuple = env_config["action_to_tuple"]
            self.end_game = len(self._action_to_tuple) - 1
            self.action_space = spaces.Discrete(len(self._action_to_tuple))
            self._use_hierarchical = True
            from inst_space import tuple_to_instruction
            self._tuple_to_instruction = tuple_to_instruction
            self.instruction_space = [
                Instruction("mov", False, "", False)
                .add_op(RegisterOp("rax", 64, False, True))
                .add_op(RegisterOp("rbx", 64, True, False))
            ]
        else:
            self.instruction_space = env_config["instruction_space"]
            self.end_game = len(self.instruction_space)
            self.action_space = spaces.Discrete(len(self.instruction_space) + 1)
            self._use_hierarchical = False

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
                "instruction": spaces.Box(low = -1, high = len(self._action_to_tuple) + 1, shape = (self.seq_size, 4), dtype=np.int64),
                # "htrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.max_trace_len), dtype=np.int64),
                # "ctrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.max_trace_len), dtype=np.int64),
                "htrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.num_inputs), dtype=np.int64),
                "ctrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.num_inputs), dtype=np.int64),
                "recovery_cycles": spaces.Box(low = -1, high = np.iinfo(np.int32).max, shape = (self.seq_size, self.num_inputs), dtype=np.int64),
                "transient_uops": spaces.Box(low = -1, high = np.iinfo(np.int32).max, shape = (self.seq_size, self.num_inputs), dtype=np.int64)
            }
        )
        print(f"observation space: {self.observation_space}")

        # initialize Printer, Program, Executor, Model, Analyzer, Input Generator
        target_desc = X86TargetDesc()
        self.printer = _X86Printer(target_desc)
        instruction_set = InstructionSet("/home/hz25d/sca-fuzzer/base.json")
        unique_instruction_names = set(spec.name.lower() for spec in instruction_set.instructions)
        # self.opcode_vocab = list(dict.fromkeys(inst.name.lower() for inst in self.instruction_space))
        self.opcode_vocab = list(unique_instruction_names)
        # self.reg_vocab = instruction_set.get_reg64_spec()
        self.reg_vocab = sorted(list(instruction_set.get_reg64_spec()))
        # print("opcode_vocab", self.opcode_vocab)

        self.asm_parser = X86AsmParser(instruction_set, target_desc)
        self.elf_parser = ELFParser(target_desc)
        self.generator = X86Generator(seed=CONF.program_generator_seed, instruction_set=instruction_set, target_desc=target_desc, asm_parser=self.asm_parser, \
                                      elf_parser=self.elf_parser)
        self.generator.instruction_space = self.instruction_space
        self.new_program = self.generator.create_test_case_SpecRL("/home/hz25d/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/my_test_case.asm", disable_assembler=True, generate_empty_case=True)
        self.base_program = deepcopy(self.new_program)
        self.executor = X86IntelExecutor()
        self.executor.valid_mem_base = 0x0
        self.executor.valid_mem_limit = 0x100000
        self.addr_mask = self.executor.valid_mem_limit - 1

        self.model = factory.get_model(self.executor.read_base_addresses())
        print(f"\nSandbox Base Address and Code Base Address (base 10): {self.executor.read_base_addresses()}\n")
        self.analyser = factory.get_analyser()
        self.input_gen = factory.get_data_generator(CONF.data_generator_seed)
        self.inputs = self.input_gen.generate(self.num_inputs, n_actors=1) # at some point would like these inputs to fall under action space

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
        end = action == self.end_game
        truncate = self.num_steps >= self.max_steps
        if not end and not truncate:
            if self._use_hierarchical:
                o, rs, rd, imm = self._action_to_tuple[action]
                inst_action = self._tuple_to_instruction(o, rs, rd, imm)
                # if inst_action is None:
                #     step_obs = self._get_obs()
                #     return step_obs, -20, end, truncate, {"program": self.new_program}
            else:
                inst_action = self.instruction_space[action]

            self.step_counter += 1
            print(f"NUMBER OF STEPS: {self.step_counter}")

            target_desc = X86TargetDesc()
            self.generator._insert_bb_index = random.choice([0, 1])
            passed_inst = X86CheckAll(self.generator, self.new_program, inst_action, target_desc)
            # passed_loop = self._infiniteLoopCheck(self.new_program, inst_action, 1)
            passed_loop = True
            if not passed_inst:
                print("DIDN'T PASS INSTRUCTION CHECK, NOT A VALID INSTRUCTION, THROWING AWAY")
                step_obs = self._get_obs()
                step_reward = -20
                return step_obs, step_reward, end, truncate, {"program": self.new_program}
            elif not passed_loop:
                print("DIDN'T PASS INFINITE LOOP CHECK, THROWING AWAY STEP")
                step_obs = self._get_obs()
                step_reward = -20
                return step_obs, step_reward, end, truncate, {"program": self.new_program}
            else:
                self.generator.insert_instruction_in_test_case_randomly(self.new_program, inst_action, self.generator._insert_bb_index)
                print(f"adding step {inst_action}")
                self.num_steps += 1
                self.succ_step_counter += 1
                print(f"NUMBER OF SUCCESSFUL STEPS: {self.succ_step_counter}")
            self.generator._insert_bb_index = None
        print()
        print("#=======================================================#")
        print("program: ")
        step_obs = self._get_obs()
        step_reward = self._reward()
        print(f"reward: {step_reward}")
        print("#=======================================================#")
        return step_obs, step_reward, end, truncate, {"program": self.new_program}


    """
    reset():
    returns an "initial observation", which in this case is blank

    clears program, resets num_steps
    note that resetting uarch state is done at observation in _get_obs
    """
    def reset(self, seed, options):
        print("#=======================================================#")
        print("#                     RESETTING                         #")
        print("#=======================================================#")
        super().reset(seed=seed)
        self.counter += 1
        print(f"NUMBER OF TEST CASES: {self.counter}")
        self.num_steps = 0
        self.bad_case = False
        self.new_program = self.generator.create_test_case_SpecRL("/home/hz25d/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/my_test_case.asm", disable_assembler=True, generate_empty_case=True)

        return (self._get_obs(), {"program": self.new_program})

    # extra functions that could be used down the line to visualize the env
    def render(self):
        return

    def close(self):
        return


    """
    _get_obs():
    returns observation, a dictionary of nparrays

    _get_obs breaks the program down into inst_1, inst_1 + inst_2, inst_1 + inst_2 + inst_3...
    in order to discern impact of each instruction
    calls _obs_program to get the actual observation for each iteration
    """
    def _get_obs(self):
        obs = {
            # "instruction": np.full((self.seq_size,), -1, dtype=np.int64),
            "instruction": np.full((self.seq_size, 4), -1, dtype=np.int64),
            "htrace": np.full((self.seq_size, self.num_inputs), 0, dtype=np.int64),
            "ctrace": np.full((self.seq_size, self.num_inputs), 0, dtype=np.uint64),
            "recovery_cycles": np.full((self.seq_size, self.num_inputs), -1, dtype=np.int64),
            "transient_uops": np.full((self.seq_size, self.num_inputs), -1, dtype=np.int64)
        } # filled with -1's as filler

        # temp file management
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_path:

            os.chdir(temp_path)
            count = 0 # iteration counter, necessary as some instructions in are instrumentation

            all_instructions = []
            for bb in self.new_program.iter_basic_blocks():
                for instr in bb:
                    new_instr = copy.deepcopy(instr)
                    new_instr._section_id = -1
                    all_instructions.append(new_instr)
            total_instructions_num = len(all_instructions)

            for i in range(1, total_instructions_num): # 1 to account for added instrumentation at start
                target_inst = all_instructions[i]
                if target_inst.is_instrumentation:
                    continue
                count += 1

                # temp program / file creation
                temp_asm_path = f"temp_obs_{i}.asm"
                temp_bin_path = f"temp_obs_{i}.o"
                os.makedirs("/home/hz25d/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/debug_asm", exist_ok=True)
                # we can not create new test cases here, cuz the instruments are randomly generated.
                temp_program = self.generator.create_asm_with_existing_test_case(temp_asm_path, self.base_program)
                for j in range(1, i + 1):
                    self.generator.insert_instruction_in_test_case(temp_program, all_instructions[j])
                    all_instructions[j]._section_id = -1
                    
                # shutil.copyfile(temp_asm_path, f"/home/hz25d/sca-fuzzer/rvzr/SpecRL/debug_asm/temp_obs_{j}.asm")

                temp_program.assign_obj(temp_bin_path)
                assemble(temp_program)
                self.elf_parser.populate_elf_data(temp_program.get_obj(), temp_program)

                temp_obs = self._obs_program(temp_program)

                print(f"\niteration {count} observations: ")

                # fill appropriate observation row in
                # obs["instruction"][count - 1] = temp_obs[0]
                obs["instruction"][count - 1] = np.array(temp_obs[0])
                # htrace/ctrace: (num_inputs,) per step, stored in (seq_size, num_inputs) array
                temp_htrace = np.array(temp_obs[1], dtype=np.int64)
                obs["htrace"][count - 1, : temp_htrace.shape[0]] = temp_htrace[: self.num_inputs]

                temp_ctrace = np.array(temp_obs[2], dtype=np.uint64)
                obs["ctrace"][count - 1, : temp_ctrace.shape[0]] = temp_ctrace[: self.num_inputs]

                obs["recovery_cycles"][count - 1] = temp_obs[3]
                obs["transient_uops"][count - 1] = temp_obs[4]

        # temp file cleanup
        os.chdir(cwd)
        return obs


    """
    _obs_program(program):
    returns (instruction index, htrace, ctrace, max recovery cycles, max transient uoperations)

    _obs_program is the function that actually executes the program
    uses revizor's executor and model code to observe the program
    """
    def _obs_program(self, program: TestCaseProgram):
        self.executor.load_test_case(program)
        self.model.load_test_case(program)

        ctraces = self.model.trace_test_case(self.inputs, 1)
        ctraces_obs = [ct.__hash__() for ct in ctraces]

        # clamp memory operands to safe region
        self.executor.valid_mem_base = 0x0
        self.executor.valid_mem_limit = 0x100000

        htraces = self.executor.trace_test_case(self.inputs, n_reps=1)

        n_reps = 1
        threshold = n_reps // 10 if n_reps >= 10 else 1
        htraces_obs = []
        for htrace in htraces:
            raw_samples = htrace.get_raw_traces()
            counter = Counter(raw_samples)
            merged_trace = 0
            for trace_val, count in counter.items():
                if count > threshold:
                    merged_trace |= int(trace_val)
            htraces_obs.append(merged_trace)

        #pfc_feedback = self.executor.get_last_feedback()
        recovery = []
        transient = []
        pfc_feedback = [ht.get_max_pfc() for ht in htraces]
        for _, pfc_values in enumerate(pfc_feedback):
            recovery.append(pfc_values[2])
            if (pfc_values[0] > pfc_values[1]):
                transient.append(pfc_values[0] - pfc_values[1])
            else: transient.append(0)
        last_instr_info = tuple(self._extract_last_instr_info(program))

        return (last_instr_info, htraces_obs, ctraces_obs, recovery, transient)


    """
    reward():
    returns a reward for the agent

    Positive Rewards:
    -Speculative Leak
    -Misspeculation
    -Observable misspeculation
    Negative Rewards:
    -# of instructions
    -diverseness of instruction sequence (not implemented)

    rewards are calculated by calling _full_obs_program
    """
    def _reward(self):
        reward = 0
        self._full_obs_program(self.new_program, self.inputs) # this is where the analyzer, observation filter, etc... are called
        if self.leak:
            print("\n!!! LEAK OCCURED !!!\n")
            reward += 9999 # reward for leak
        reward = reward + 30 if self.misspec else reward - 30 # reward/punish for passing/failing misspeculative filters
        print(f"misspec: {self.misspec}, reward: {reward}")
        reward = reward + 50 if self.observable else reward - 50 # reward/punish for passing/failing observation filters
        print(f"observable: {self.observable}, reward: {reward}")
        reward -= 1 # negative reward for each additional step
        print(f"step penalty, reward: {reward}")

        return reward


    """
    _full_obs_program(program, inputs)
    sets speculative leak, misspeculation, observable speculation flags

    essentially has the same functionality as _obs_program except it it checks for a leak and observable speculation
    pulled from revizor code
    """
    def _full_obs_program(self, program: TestCaseProgram, inputs: List[InputData]) -> bool:
        self.leak = False
        self.misspec = False
        self.observable = False

        self.fuzzer = X86Fuzzer("/home/hz25d/sca-fuzzer/base.json", os.getcwd(), existing_test_case= "/home/hz25d/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/my_test_case.asm", input_paths=self.inputs)
        self.fuzzer.model = self.model
        self.fuzzer.data_gen = self.input_gen
        self.fuzzer.analyser = self.analyser
        self.fuzzer.executor = self.executor
        asm = tempfile.NamedTemporaryFile(delete=False)
        bin = tempfile.NamedTemporaryFile(delete=False)

        temp = copy.deepcopy(program)

        all_instructions = []
        for bb in temp.iter_basic_blocks():
            for instr in list(bb):
                instr._section_id = -1
                all_instructions.append(instr)
        total_instructions_num = len(all_instructions)
        print("total instructions in program: ", total_instructions_num)
        print("all instructions: ", [self.printer._instruction_to_str(instr) for instr in all_instructions])


        temp.assign_obj(bin.name)
        assemble(temp)
        self.elf_parser.populate_elf_data(temp.get_obj(), temp)


        manager = _RoundManager(self.fuzzer, temp, inputs)
        manager._boost_inputs()
        boosted_inputs = manager.boosted_inputs

        # check for violations
        ctraces = self.model.trace_test_case(boosted_inputs, 1)
        htraces = self.executor.trace_test_case(boosted_inputs, 10)

        # check if misspec occurs, updates flag
        pfc_feedback = [ht.get_max_pfc() for ht in htraces]
        for i, pfc_values in enumerate(pfc_feedback):
            if pfc_values[0] > pfc_values[1] or pfc_values[2] > 0:
                self.misspec = True

        # check if it's observable, updates flag

        fenced = tempfile.NamedTemporaryFile(delete=False)
        fenced_obj = tempfile.NamedTemporaryFile(delete=False)

        debug_path = "/home/hz25d/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/debug_asm"
        os.makedirs(debug_path, exist_ok=True)
        fenced_test_case = _create_fenced_test_case(temp._asm_path, fenced.name, self.asm_parser, self.generator, self.elf_parser)

        self.executor.load_test_case(fenced_test_case)
        fenced_htraces = self.executor.trace_test_case(inputs, n_reps=10)


        if fenced_htraces != htraces:
            self.observable = True

        traces_match = True
        for i, _ in enumerate(inputs):
            if not self.analyser.htraces_are_equivalent(fenced_htraces[i], htraces[i]):
                traces_match = False
                break
        # print("traces_match ***************", traces_match)

        violations = self.fuzzer.start_SpecRL(1, len(inputs), 0, False, False, type_='asm')
        if not violations:  # nothing detected? -> we are done here, move to next test case
            return None

        print("\n\n\nFOUND VIOLATION!!!\n\n\n")
        os.remove(fenced.name)
        os.remove(fenced_obj.name)
        os.remove(asm.name)
        os.remove(bin.name)

        # 2. Repeat with with max nesting
        if 'seq' not in CONF.contract_execution_clause:
            # self.LOG.fuzzer_nesting_increased()
            #boosted_inputs = fuzzer.boost_inputs(inputs, CONF.model_max_nesting)
            manager = _RoundManager(self.fuzzer, temp, inputs)
            boosted_inputs = manager._boost_inputs()
            ctraces = self.model.trace_test_case(boosted_inputs, CONF.model_max_nesting)
            htraces = self.executor.trace_test_case(boosted_inputs, CONF.executor_repetitions)
            violations = self.fuzzer.analyser.filter_violations(boosted_inputs, ctraces, htraces, True)
            if not violations:
                print("\n\n\n FAILED MAX NESTING \n\n\n")
                return None

        # 3. Check if the violation is reproducible
        # if self.fuzzer.check_if_reproducible(violations, boosted_inputs, htraces):
        #     # STAT.flaky_violations += 1
        #     if CONF.ignore_flaky_violations:
        #         print("\n\n\n FAILED REPRODUCIBLE \n\n\n")
        #         return None

        # # 4. Check if the violation survives priming
        # if not CONF.enable_priming:
        #     return violations[-1]
        # # STAT.required_priming += 1

        # violation_stack = list(violations)  # make a copy
        # while violation_stack:
        #     # self.LOG.fuzzer_priming(len(violation_stack))
        #     violation: EquivalenceClass = violation_stack.pop()
        #     if self.fuzzer.priming(violation, boosted_inputs):
        #         break
        # else:
        #     # All violations were cleared by priming.
        #     print("\n\n\n FAILED PRIMING \n\n\n")
        #     return None

        print("\n\n\nFOUND VIOLATION!!!\n\n\n")

        # Violation survived priming. Report it
        self.leak = True
        return violations

    def _infiniteLoopCheck(self, prog: TestCaseProgram, instr: Instruction, timeout: int) -> bool:
        prog_ = copy.deepcopy(prog)
        unique = uuid.uuid4().hex
        self.generator.insert_instruction_in_test_case(prog_, instr)
        temp_asm_path = f"/tmp/test_case_{unique}.s"
        temp_bin_path = f"/tmp/test_case_{unique}.o"
        prog_._asm_path = temp_asm_path
        prog_.assign_obj(temp_bin_path)
        self.printer.print(prog_)
        assemble(prog_)
        self.elf_parser.populate_elf_data(prog_.get_obj(), prog_)
        self.model.load_test_case(prog_)


        return self.model.check_inf_loop(self.inputs, 1, timeout)

    def aggregate_htraces(self, htraces: List[HTrace], n_reps: int) -> List[int]:

        threshold = n_reps // 10 if n_reps >= 10 else 0
        aggregated_traces = []

        for htrace in htraces:
            raw_samples = htrace.get_raw_traces()
            counter = Counter(raw_samples)
            merged_trace = 0
            for trace_val, count in counter.items():
                if count > threshold:
                    merged_trace |= int(trace_val)
            aggregated_traces.append(merged_trace)

        return aggregated_traces

    def apply_selective_fencing(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        fenced_lines = []
        in_test_zone = False
        for line in lines:
            stripped = line.strip()
            if ".line_1:" in stripped:
                in_test_zone = True
            if ".exit_0:" in stripped or ".macro.measurement_end" in stripped:
                in_test_zone = False
            fenced_lines.append(line)
            if in_test_zone and stripped:
                fenced_lines.append("lfence\n")
        with open(file_path, 'w') as f:
            f.writelines(fenced_lines)

    def _get_last_instruction(self, program: TestCaseProgram) -> Optional[Instruction]:
        """Get the last instruction in program order (excluding macros)."""
        last_instr = None
        for bb in program.iter_basic_blocks():
            for instr in bb:
                if instr.name != "macro":
                    last_instr = instr
        return last_instr

    def _normalize_reg_name(self, reg_name: str) -> str:
        mapping = {
                'eax': 'rax', 'ax': 'rax', 'al': 'rax', 'ah': 'rax',
                'ebx': 'rbx', 'bx': 'rbx', 'bl': 'rbx', 'bh': 'rbx',
                'ecx': 'rcx', 'cx': 'rcx', 'cl': 'rcx', 'ch': 'rcx',
                'edx': 'rdx', 'dx': 'rdx', 'dl': 'rdx', 'dh': 'rdx',
                'esi': 'rsi', 'si': 'rsi', 'sil': 'rsi',
                'edi': 'rdi', 'di': 'rdi', 'dil': 'rdi',
                'rip': 'rip',
            }
        return mapping.get(reg_name, reg_name)
    
    def _extract_last_instr_info(self, program: TestCaseProgram) -> List[int]:
        """Return [opname_id, reg_src_id, reg_dst_id, imm_id] for last instruction."""
        last_instr = self._get_last_instruction(program)
        if last_instr is None:
            return [-1, -1, -1, -1]

        name_lower = last_instr.name.lower()
        opname_id = self.opcode_vocab.index(name_lower) if name_lower in self.opcode_vocab else -1

        reg_ops = last_instr.get_reg_operands(include_implicit=True)
        reg_src_id = -1
        reg_dst_id = -1

        for op in reg_ops:
            val_raw = op.value.lower()
            val_norm = self._normalize_reg_name(val_raw)
            if val_norm in self.reg_vocab:
                idx = self.reg_vocab.index(val_norm)
                if op.src and reg_src_id == -1:
                    reg_src_id = idx
                if op.dest and reg_dst_id == -1:
                    reg_dst_id = idx

        imm_ops = last_instr.get_imm_operands()
        imm_id = 0 if imm_ops else -1

        return [opname_id, reg_src_id, reg_dst_id, imm_id]

    def _get_opcode_size(self):
        return len(self.opcode_vocab)   

    def _get_reg_size(self):
        return len(self.reg_vocab)
