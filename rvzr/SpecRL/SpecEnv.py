import numpy as np
import subprocess
import gymnasium as gym
import uuid
from gymnasium import spaces
from typing import Counter, List, Optional
from rvzr.code_generator import Printer
from rvzr.arch.x86.generator import _X86Printer
from rvzr.arch.x86.generator import newPrinter
from rvzr.tc_components.test_case_code import Program, TestCaseProgram
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
from rvzr.tc_components.instruction import Instruction
from rvzr.arch.x86.executor import X86IntelExecutor
from rvzr.config import CONF
from rvzr.code_generator import assemble
from rvzr.code_generator import map_address
from rvzr.arch.x86.target_desc import X86TargetDesc
from check import X86CheckAll
from rvzr.traces import CTrace
from rvzr.traces import HTrace
from rvzr import factory
from rvzr.fuzzer import Fuzzer, _RoundManager
from rvzr.arch.x86.fuzzer import X86Fuzzer
from rvzr.arch.x86.fuzzer import _create_fenced_test_case
from interfaces import Measurement, EquivalenceClass
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
        target_desc = X86TargetDesc()
        # self.printer = newPrinter(target_desc) # using x86 printer for now, may need to change later
        self.printer = _X86Printer(target_desc) 
        #self.program = Program(self.seq_size, self.asm_path, self.bin_path) #Initialization may need to pass in more args later compare to orignal SpecEnv
        # Todo: Need to use code_generator to generate program.
        instruction_set = InstructionSet("/home/hz25d/sca-fuzzer/base.json")
        self.asm_parser = X86AsmParser(instruction_set, target_desc)
        self.elf_parser = ELFParser(target_desc)
        self.generator = X86Generator(seed=CONF.program_generator_seed, instruction_set=instruction_set, target_desc=target_desc, asm_parser=self.asm_parser, \
                                      elf_parser=self.elf_parser)
        self.generator.instruction_space = self.instruction_space
        self.new_program = self.generator.create_test_case("/home/hz25d/sca-fuzzer/rvzr/SpecRL/my_test_case.asm", disable_assembler=True, generate_empty_case=True)
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
        end = (action == self.end_game)
        truncate = (self.num_steps >= self.max_steps)
        if not end and not truncate:
            inst_action = self.instruction_space[action]

            self.step_counter += 1
            print(f"NUMBER OF STEPS: {self.step_counter}")

            # run checks / instrument
            target_desc = X86TargetDesc()
            passed_inst = X86CheckAll(self.generator, self.new_program, inst_action, target_desc)
            # passed_loop = self._infiniteLoopCheck(self.program, inst_action, 1)
            # passed_inst = True
            passed_loop = True
            if (not passed_inst):
                print("DIDN'T PASS INSTRUCTION CHECK, NOT A VALID INSTRUCTION, THROWING AWAY")
                step_obs = self._get_obs()
                step_reward = -20
                return (step_obs, step_reward, end, truncate, {"program": self.new_program})
            elif (not passed_loop):
                print("DIDN'T PASS INFINITE LOOP CHECK, THROWING AWAY STEP")
                step_obs = self._get_obs()
                step_reward = -20
                return (step_obs, step_reward, end, truncate, {"program": self.new_program})
            else:
                # self.program.append(self.instruction_space[action])
                self.generator.insert_instruction_in_test_case(self.new_program, self.instruction_space[action])
                print(f"adding step {self.instruction_space[action]}")
                self.succ_step_counter += 1
                print(f"NUMBER OF SUCCESSFUL STEPS: {self.succ_step_counter}")
        print()
        print("#=======================================================#")
        print("program: ")
        # self.program.print()
        step_obs = self._get_obs()
        step_reward = self._reward()
        print(f"reward: {step_reward}")
        print("#=======================================================#")
        return (step_obs, step_reward, end, truncate, {"program": self.new_program})


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
        # self.program = Program(self.seq_size, self.asm_path, self.bin_path)
        self.new_program = self.generator.create_test_case("/home/hz25d/sca-fuzzer/rvzr/SpecRL/my_test_case.asm", disable_assembler=True, generate_empty_case=True, \
                                                           instruction_space=self.generator.instruction_space)
        # all_instructions = []
        # for bb in self.new_program.iter_basic_blocks():
        #     for instr in bb:
        #         all_instructions.append(instr)
        # total_instructions_num = len(all_instructions)
        # print("new_program after reset: ", total_instructions_num)
        # print(f"bin path: {self.program.bin_path}")
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
            "instruction": np.full((self.seq_size,), -1, dtype=np.int64),
            "htrace": np.full((self.seq_size, self.max_trace_len), -1, dtype=np.int64),
            "ctrace": np.full((self.seq_size, self.max_trace_len), -1, dtype=np.int64),
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
            # count = total_instructions_num - 2
            # print("count = ", count)
            # print([self.printer._instruction_to_str(instr) for instr in all_instructions])
            # for i in range(self.program.length):
            for i in range(1, total_instructions_num): # -1 to account for added instrumentation at start
                target_inst = all_instructions[i] # +1 to account for added instrumentation at start
                # check if program[i] is instrumentation. If so, skip
                # if (self.program.getInd(i).is_instrumentation):
                #     continue
                if target_inst.is_instrumentation:
                    continue
                count += 1

                # temp program / file creation
                temp_asm_path = f"temp_obs_{i}.asm"
                temp_bin_path = f"temp_obs_{i}.o"
                os.makedirs("/home/hz25d/sca-fuzzer/rvzr/SpecRL/debug_asm", exist_ok=True)
                # temp = Program(i + 1, temp_asm_path, temp_bin_path)
                temp_program = self.generator.create_test_case(temp_asm_path, disable_assembler=True, generate_empty_case=True, instruction_space=self.generator.instruction_space)
                # print(f"temp program before adding instructions: {temp_program}", temp_program.asm_path())

                for j in range(1, i + 1):
                    self.generator.insert_instruction_in_test_case(temp_program, all_instructions[j])
                    all_instructions[j]._section_id = -1
                    # print("i=", i, "j=", j)
                    # shutil.copyfile(temp_asm_path, f"/home/hz25d/sca-fuzzer/rvzr/SpecRL/debug_asm/temp_obs_{j}.asm")
                # for _ in range(i + 1):
                #     temp.append(curr)
                #     curr = curr.next
                # self.printer.add_line_num(temp_program)
                temp_program.assign_obj(temp_bin_path)
                assemble(temp_program)
                self.elf_parser.populate_elf_data(temp_program.get_obj(), temp_program)
                # map_address(temp_program, temp_bin_path)

                # print(f"Checking object file: {temp_program.get_obj().obj_path}")
                # if not os.path.exists(temp_program.get_obj().obj_path) or os.path.getsize(temp_program.get_obj().obj_path) == 0:
                #     print("CRITICAL: Object file is missing or empty!")
                # else:
                #     print("Object file exists and is non-empty.")
                
                # print(f"mapped addresses: {temp_program.address_map}")
                # self.printer.map_addresses(temp, temp_bin_path)
                # self.printer.create_pte(temp)



                #subprocess.run("/home/laievan/specenv/SpecRL/src/reset_branch") # want to run reset_branch between iterations but not between inputs
                temp_obs = self._obs_program(temp_program)

                print(f"\niteration {count} observations: ")
                #print(temp_obs)

                # fill appropriate observation row in
                obs["instruction"][count - 1] = temp_obs[0]

                temp_htrace = np.array(temp_obs[1]) # some extra work needed to pad in order to fit the shape
                padded_htrace = np.full((self.max_trace_len,), -1, dtype = temp_htrace.dtype)
                padded_htrace[:temp_htrace.shape[0]] = temp_htrace
                obs["htrace"][count - 1] = padded_htrace

                temp_ctrace = np.array(temp_obs[2])
                padded_ctrace = np.full((self.max_trace_len,), -1, dtype = temp_ctrace.dtype)
                padded_ctrace[:temp_ctrace.shape[0]] = temp_ctrace
                obs["ctrace"][count - 1] = padded_ctrace

                print("temp htrace: ", temp_htrace)
                print("temp ctrace: ", temp_ctrace)

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
        # self.executor.load_program(program)
        # self.model.load_program(program)
        self.executor.load_test_case(program)
        self.model.load_test_case(program)

        # could boost inputs here?
        ctraces = self.model.trace_test_case(self.inputs, 1)
        #raw_ctraces = [ct.get_untyped() for ct in ctraces]  
        # clamp memory operands to safe region
        self.executor.valid_mem_base = 0x0
        self.executor.valid_mem_limit = 0x100000

        htraces = self.executor.trace_test_case(self.inputs, n_reps=1)
        htraces_int = self.aggregate_htraces(htraces, n_reps=1)
        #raw_htraces = [ht.get_raw_traces() for ht in htraces]

        #pfc_feedback = self.executor.get_last_feedback()
        recovery = []
        transient = []
        pfc_feedback = [ht.get_max_pfc() for ht in htraces]
        for i, pfc_values in enumerate(pfc_feedback):
            recovery.append(pfc_values[2])
            if (pfc_values[0] > pfc_values[1]):
                transient.append(pfc_values[0] - pfc_values[1])
            else: transient.append(0)
        return (program.__len__(), htraces_int, ctraces, recovery, transient)
        #return (program.__len__(), htraces.get_raw_traces(), ctraces.get_untyped(), recovery, transient)


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
            exit()
            reward += 9999 # reward for leak
        reward = reward + 30 if self.misspec else reward - 30 # reward/punish for passing/failing misspeculative filters
        reward = reward + 50 if self.observable else reward - 50 # reward/punish for passing/failing observation filters
        reward -= 1 # negative reward for each additional step

        return reward


    """
    _full_obs_program(program, inputs)
    sets speculative leak, misspeculation, observable speculation flags

    essentially has the same functionality as _obs_program except it it checks for a leak and observable speculation
    pulled from revizor code
    """
    def _full_obs_program(self, program: TestCaseProgram, inputs: List[InputData]) -> Optional[EquivalenceClass]:
        self.leak = False
        self.misspec = False
        self.observable = False

        self.fuzzer = X86Fuzzer("/home/hz25d/sca-fuzzer/base.json", os.getcwd(), input_paths=inputs)
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

        # self.printer.add_line_num(temp)
        # self.printer.print(temp, lines=15)
        temp.assign_obj(bin.name)
        assemble(temp)
        self.elf_parser.populate_elf_data(temp.get_obj(), temp)
        # shutil.copyfile(temp.asm_path(), f"/home/hz25d/sca-fuzzer/rvzr/SpecRL/debug_asm/temp_full_obs.asm")
        # map_address(temp, temp.get_obj().obj_path)
        #self.printer.map_addresses(temp, temp.bin_path)
        
        self.executor.load_test_case(temp)
        self.model.load_test_case(temp)

        ctraces: List[CTrace]
        htraces: List[HTrace]

        # at this point we need to increase the effectiveness of inputs
        # so that we can detect contract violations (note that it wasn't necessary
        # up to this point because we weren't testing against a contract)
        #boosted_inputs: List[InputData] = fuzzer.generate_boosted(inputs, 1)
        manager = _RoundManager(self.fuzzer, temp, self.inputs)
        manager._boost_inputs()
        boosted_inputs = manager.boosted_inputs

        # check for violations
        ctraces = self.model.trace_test_case(boosted_inputs, 1)
        htraces = self.executor.trace_test_case(boosted_inputs, CONF.executor_repetitions)

        # check if misspec occurs, updates flag
        pfc_feedback = [ht.get_max_pfc() for ht in htraces]
        # pfc_feedback = self.executor.get_last_feedback()
        for i, pfc_values in enumerate(pfc_feedback):
            if pfc_values[0] > pfc_values[1] or pfc_values[2] > 0:
                self.misspec = True

        # check if it's observable, updates flag
  
        fenced = tempfile.NamedTemporaryFile(delete=False)
        fenced_obj = tempfile.NamedTemporaryFile(delete=False)
        fenced_test_case = self.generator.create_test_case(fenced.name, disable_assembler=True, generate_empty_case=True, instruction_space=self.generator.instruction_space)
        

        debug_path = "/home/hz25d/sca-fuzzer/rvzr/SpecRL/debug_asm"
        os.makedirs(debug_path, exist_ok=True)
        
        for i in range(1, total_instructions_num):
            dest_path = f"{debug_path}/fenced_obs{i}.asm"
            self.generator.insert_instruction_in_test_case(fenced_test_case, all_instructions[i])
            self.generator.insert_instruction_in_test_case(fenced_test_case, Instruction("lfence"))

        self.printer.add_line_num_full_obs(fenced_test_case)
        # self.apply_selective_fencing(fenced_test_case.asm_path())
        # run('awk \'//{print $0, "\\nlfence"}\' ' + temp.asm_path() + '>' + fenced.name, shell=True)
        fenced_test_case.assign_obj(fenced_obj.name)
        assemble(fenced_test_case)


        self.elf_parser.populate_elf_data(fenced_test_case.get_obj(), fenced_test_case)
        shutil.copyfile(fenced.name, dest_path)

        # run('awk \'//{print $0, "\\nlfence"}\' ' + temp.asm_path() + '>' + fenced.name, shell=True)
        # assemble(fenced.name)
        # fenced_test_case = _create_fenced_test_case(temp._asm_path, fenced.name, self.asm_parser, self.generator,self.elf_parser)

        self.executor.load_test_case(fenced_test_case)
        fenced_htraces = self.executor.trace_test_case(inputs, n_reps=1)


        # fenced_test_case = self.generator.create_test_case(fenced.name, disable_assembler=False, generate_empty_case=False, instruction_space=self.generator.instruction_space)
        # fenced_test_case._obj = fenced_obj.name
        # self.executor.load_test_case(fenced_test_case)
        # self.executor.valid_mem_base = 0x0
        # self.executor.valid_mem_limit = 0x100000
        # fenced_htraces = self.executor.trace_test_case(inputs, n_reps=1)
        os.remove(fenced.name)
        os.remove(fenced_obj.name)
        os.remove(asm.name)
        os.remove(bin.name)

        # program._asm_path = self.asm_path
        # program._obj = self.bin_path

        if fenced_htraces != htraces:
            self.observable = True

        # self.LOG.trc_fuzzer_dump_traces(self.model, boosted_inputs, htraces, ctraces,
        #                                 self.executor.get_last_feedback())
        # violations = self.fuzzer.analyser.filter_violations(boosted_inputs, ctraces, htraces, True)
        violations = self.fuzzer.analyser.filter_violations(ctraces, htraces, temp, boosted_inputs, True)
        if not violations:  # nothing detected? -> we are done here, move to next test case
            return None

        print("\n\n\nFOUND VIOLATION!!!\n\n\n")

        # 2. Repeat with with max nesting
        if 'seq' not in CONF.contract_execution_clause:
            # self.LOG.fuzzer_nesting_increased()
            #boosted_inputs = fuzzer.boost_inputs(inputs, CONF.model_max_nesting)
            boosted_inputs = _RoundManager._boost_inputs()
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
        self.printer.add_line_num(prog_)
        prog_.asm_path = f"/tmp/test_case_{unique}.s"
        prog_.assign_obj(f"/tmp/test_case_{unique}.o")

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




