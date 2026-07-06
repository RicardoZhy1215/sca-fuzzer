import random

import numpy as np
import subprocess
import gymnasium as gym
import uuid
import re
import ctypes
from gymnasium import spaces
from copy import deepcopy
from dataclasses import dataclass
from typing import Counter, Dict, FrozenSet, List, Optional, Sequence, Tuple
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
from datetime import datetime
from subprocess import run, CalledProcessError
from inst_space import OPERAND_SPACE, DST_REGS_SPACE, SRC_REGS_SPACE, IMMS_SPACE
from pattern_matching import VulnerabilityPatternMatcher, PatternToken


# Speculative Store Bypass prctl constants. Originally V4-only, but the
# `_force_kernel_ssb_vulnerable()` hook below is required for Ray rollout
# workers regardless of vulnerability_type — without it the workers boot
# with CONF.x86_executor_enable_ssbp_patch=True (module default) and the
# executor re-enables SSBD MSR, silently disabling v4 even if the agent
# happens to produce a v4 gadget. Keep these enabled at all times.
PR_SET_SPECULATION_CTRL = 53
PR_SPEC_STORE_BYPASS = 0
PR_SPEC_ENABLE = 1 << 1



_EMBEDDER_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}


def _get_text_embedder(model_name: str, device: str):
    key = (model_name, device)
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]
    from transformers import AutoTokenizer, AutoModel  # lazy import
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    _EMBEDDER_CACHE[key] = (tok, model)
    return tok, model


def _filter_reference_asm_text(path: str) -> str:
    """
    Extract just the agent-emitted instructions from a reference .asm file.
    Skips:
        - directives / labels (lines starting with '.')
        - lines tagged '# instrumentation' (sandbox-inserted ops like
          `and rax, 0b1111111111111`)
        - pure comments
    Trailing inline comments are stripped. The output is a newline-joined
    string of one instruction per line, matching the format we'll generate
    for SpecRL programs.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("."):
                continue
            if "# instrumentation" in s:
                continue
            if s.startswith("#"):
                continue
            s = s.split("#", 1)[0].strip()
            if s:
                out.append(s)
    return "\n".join(out)


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
    end_game_counter: int

    bin_path = "my_test_case.o"
    asm_path = "my_test_case.asm"

    def __init__(self, env_config):
        # Per-process SSB opt-in. Originally V4-only but kept on at all times:
        # Ray rollout workers re-import CONF fresh, so the only reliable place
        # to switch SSBD off is here, inside the worker's __init__.
        self._ensure_ssb_vulnerable_for_process()
        # Same reason as SSBD above: Ray rollout workers re-import rvzr.config
        # with DEFAULTS, so the `--config big_fuzz.yaml` contract loaded in the
        # driver (train.py __main__) never reaches the model built here. Without
        # this, factory.get_model() below would silently use the default
        # 'delayed-exception-handling' clause instead of bpas, which makes the
        # fuzzer flag plain Spectre-v4 as violations. Force the intended
        # ct + cond-bpas contract here, before factory.get_model() is called.
        # ['cond', 'bpas'] -> factory resolves to the 'cond-bpas' speculator
        # (Spectre-v1 branch misprediction + v4 store bypass).
        setattr(CONF, "contract_observation_clause", "ct")
        setattr(CONF, "contract_execution_clause", ["cond", "bpas"])
        print(f"[CONTRACT-CHECK][worker pid={os.getpid()}] "
              f"obs={CONF.contract_observation_clause} "
              f"exec={CONF.contract_execution_clause} "
              f"ssbp_patch={CONF.x86_executor_enable_ssbp_patch}", flush=True)
        self.seq_size = env_config["sequence_size"]
        self.num_inputs = env_config["num_inputs"]
        self.bad_case = False
        self.counter = 0
        self.step_counter = 0
        self.succ_step_counter = 0
        self.end_game_counter = 0

        # Hierarchical tuple: action = (opcode, reg_src, reg_dst, imm)
        from hi_model import get_hierarchical_action_space
        from inst_space import tuple_to_instruction
        self.action_space = get_hierarchical_action_space()
        self._tuple_to_instruction = tuple_to_instruction
        self.end_game = (0, 0, 0, 0)
        self.instruction_space = [
            Instruction("mov", False, "", False)
            .add_op(RegisterOp("rax", 64, False, True))
            .add_op(RegisterOp("rbx", 64, True, False))
        ]

        self.num_steps = 0
        self.max_steps = self.seq_size
        # Default switched from "spectre_v4" -> "spectre_v1". v4 path is kept
        # (commented) for reference but not used during training.
        self.vulnerability_type = env_config.get("vulnerability_type", "spectre_v1")
        # Bumped default from 1.0 -> 5.0 so partial structural pattern reward
        # can out-weigh the per-step obs/misspec filter penalties below and
        # drive the agent toward v4 gadget layouts before a real leak is seen.
        self.pattern_reward_scale = float(env_config.get("pattern_reward_scale", 5.0))
        self.leak_reward = float(env_config.get("leak_reward", 600.0))
        self.trace_divergence_reward_scale = float(env_config.get("trace_divergence_reward_scale", 30.0))
        # Step-level shaping knobs (expose via env_config so train.py can tune):
        self.step_penalty = float(env_config.get("step_penalty", 0.1))
        # Mask the end_game action before the agent has had a chance to lay
        # down a gadget. Picked so a minimal v4 gadget (slow producer + store
        # + bypass load + transmitter, plus a few setup mov_ri's) comfortably fits.
        self.min_steps_before_end = int(env_config.get("min_steps_before_end", 8))
        self.early_end_penalty = float(env_config.get("early_end_penalty", 50.0))
        # Revizor-style PFC misspec signal (branch mispred / machine clear).
        # [V4-only branch commented] For Spectre v4 (SSB) this signal does
        # not fire, so the v4 path used to default this to 0.0. v1 mispredicts
        # a conditional branch -> RECOVERY_CYCLES > 0, so 30.0 is the right
        # default. Restore the simple default for v1 training.
        # self.misspec_bonus = float(env_config.get(
        #     "misspec_bonus",
        #     0.0 if self.vulnerability_type == "spectre_v4" else 30.0,
        # ))
        self.misspec_bonus = float(env_config.get("misspec_bonus", 30.0))
        self.observable_bonus = float(env_config.get("observable_bonus", 50.0))
        # Single-sided observable reward (P0.2). The old symmetric form applied
        # -observable_bonus every step that was NOT observable, which compounds
        # into a huge negative floor (-50 * seq_size) and drowns the structural
        # pattern signal. Default 0.0 here so "not observable yet" stops being
        # an active punishment and only the positive +observable_bonus remains.
        self.observable_penalty = float(env_config.get("observable_penalty", 0.0))
        self.pattern_matcher = VulnerabilityPatternMatcher(self.vulnerability_type)
        self.pattern_stage_reward = 0.0
        self.pattern_match_counts = {}
        self.trace_divergence_score = 0.0
        # Per-episode running max of each pattern counter (for RLlib custom_metrics → wandb).
        self.episode_pattern_match_max: Dict[str, int] = {}
        # Snapshot taken at reset() after an episode ends; safe when callback runs after reset.
        self._last_completed_episode_pattern_match: Dict[str, int] = {}

        # Per-episode micro-architectural signal trackers (wandb -> train/leak/*).
        #   trace_div_max : max trace_divergence_score ever hit in this episode
        #   observable_any: did ANY step have observable=True
        #   leak_any      : did ANY step see a real fuzzer-reported violation
        #   steps_observable: how many steps were observable=True (density)
        self.episode_trace_div_max: float = 0.0
        self.episode_observable_any: bool = False
        self.episode_leak_any: bool = False
        self.episode_steps_observable: int = 0
        # Completed-episode snapshots for RLlib callbacks (mirrors the
        # pattern_match_max pair above so on_episode_end can read after reset).
        self._last_episode_trace_div_max: float = 0.0
        self._last_episode_observable_any: bool = False
        self._last_episode_leak_any: bool = False
        self._last_episode_steps_observable: int = 0

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
        instr_high = 500
        # NUM_ARCH_REGS: rax, rbx, rcx, rdx, rsi, rdi (matches _REG_ID_TO_NAME_X86
        # in rvzr/traces.py and DST_REGS_SPACE in inst_space.py).
        self.num_arch_regs = 6
        self.observation_space = spaces.Dict(
            {
                "instruction": spaces.Box(low = -1, high = instr_high, shape = (self.seq_size, 4), dtype=np.int64),
                # "htrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.max_trace_len), dtype=np.int64),
                # "ctrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.max_trace_len), dtype=np.int64),
                "htrace": spaces.Box(low = min_address, high = max_address, shape = (self.seq_size, self.num_inputs), dtype=np.int64),
                "ctrace": spaces.Box(low = np.iinfo(np.uint64).min, high = np.iinfo(np.uint64).max, shape = (self.seq_size, self.num_inputs), dtype=np.uint64),
                "recovery_cycles": spaces.Box(low = -1, high = np.iinfo(np.int32).max, shape = (self.seq_size, self.num_inputs), dtype=np.int64),
                "transient_uops": spaces.Box(low = -1, high = np.iinfo(np.int32).max, shape = (self.seq_size, self.num_inputs), dtype=np.int64),
                # End-of-program register-equality encoding (per input, rax..rdi).
                # 1 = this slot's value duplicates at least one other slot in
                # the same snapshot, 0 = unique. See _collect_reg_state for
                # the rationale and the loss-of-info caveat.
                "regs": spaces.Box(low = 0, high = 1, shape = (self.seq_size, self.num_inputs, self.num_arch_regs), dtype=np.uint8),
            }
        )
        print(f"observation space: {self.observation_space}")

        # initialize Printer, Program, Executor, Model, Analyzer, Input Generator
        target_desc = X86TargetDesc()
        self.printer = _X86Printer(target_desc)
        instruction_set = InstructionSet("/home/mluo/sca-fuzzer/base.json")
        # Keep observation opcode IDs aligned with the hierarchical action space.
        self.opcode_vocab = list(OPERAND_SPACE)
        self.reg_vocab = list(DST_REGS_SPACE)
        # print("opcode_vocab", self.opcode_vocab)

        self.asm_parser = X86AsmParser(instruction_set, target_desc)
        self.elf_parser = ELFParser(target_desc)
        self.generator = X86Generator(seed=CONF.program_generator_seed, instruction_set=instruction_set, target_desc=target_desc, asm_parser=self.asm_parser, \
                                      elf_parser=self.elf_parser)
        self.generator.instruction_space = self.instruction_space
        self.new_program = self.generator.create_test_case_SpecRL("/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/my_test_case.asm", disable_assembler=True, generate_empty_case=True)
        self.base_program = deepcopy(self.new_program)
        self.executor = factory.get_executor()
        self.executor.valid_mem_base = 0x0
        self.executor.valid_mem_limit = 0x100000
        self.addr_mask = self.executor.valid_mem_limit - 1

        # Second executor in arch (GPR) mode. The kernel module's
        # /sys/rvzr_executor/enable_dbg_gpr_mode flag is GLOBAL, so this
        # instance must be used carefully: every load_test_case() flips the
        # flag, and the most recent load wins for the next trace_test_case().
        # Always re-load self.executor before reading htraces to restore flag=0.
        self.arch_executor = factory.get_executor(enable_mismatch_check_mode=True)

        self.model = factory.get_model(self.executor.read_base_addresses())
        print(f"[CONTRACT-CHECK][worker pid={os.getpid()}] model built: "
              f"speculator={type(getattr(self.model, 'speculator', None)).__name__} "
              f"tracer={type(getattr(self.model, 'tracer', None)).__name__}", flush=True)
        print(f"\nSandbox Base Address and Code Base Address (base 10): {self.executor.read_base_addresses()}\n")
        self.analyser = factory.get_analyser()
        self.input_gen = factory.get_data_generator(CONF.data_generator_seed)
        self.inputs = self.input_gen.generate(self.num_inputs, n_actors=1) # at some point would like these inputs to fall under action space

        # HARD-FORCE the kernel module's SSBD patch off. Originally V4-only,
        # kept on always: Ray rollout workers re-import CONF and otherwise
        # default x86_executor_enable_ssbp_patch=True, which re-enables SSBD
        # in ring-0 test runs and silently disables v4 even if the agent
        # produces a perfectly valid v4 gadget. Harmless under v1.
        self._force_kernel_ssb_vulnerable()

        # self.similarity_model_name = env_config.get("similarity_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        # self.similarity_device = env_config.get("similarity_device", "cuda:0" if self._cuda_available() else "cpu")
        # self.similarity_top_k = int(env_config.get("similarity_top_k", 3))
        # self.similarity_reward_scale = float(env_config.get("similarity_reward_scale", 20.0))
        # self.similarity_baseline_subtract = bool(env_config.get("similarity_baseline_subtract", False))
        # self.similarity_max_length = int(env_config.get("similarity_max_length", 512))
        # similarity machinery is generic (UniXcoder cosine vs. a directory of
        # reference asm files). The hardcoded v4 example default is commented
        # out — pass `similarity_reference_dir` in env_config to enable it for
        # any vulnerability. Empty default => similarity reward is disabled.
        ref_dir = env_config.get(
            "similarity_reference_dir",
            "/home/mluo/attack_seq_generation/side-channel-fuzzer/spectre_v4_example",
        )
        self._sim_tokenizer = None
        self._sim_model = None
        self._reference_embeddings = None  # (N_ref, hidden_dim) on similarity_device, normalized
        self._reference_centroid = None    # (hidden_dim,) on similarity_device, normalized
        self.similarity_score = 0.0
        self.episode_similarity_max = 0.0
        self._last_episode_similarity_max = 0.0
        # self._init_similarity_embedder(ref_dir)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _init_similarity_embedder(self, ref_dir: str) -> None:
        """
        Lazy-load the embedder and pre-embed reference asm files.
        On any failure (model download / no refs / etc.), disable the
        similarity reward by leaving _reference_embeddings=None.
        """
        if not ref_dir or not os.path.isdir(ref_dir):
            print(f"[SIM][SpecEnv] reference_dir '{ref_dir}' not found; similarity reward disabled")
            return
        ref_paths = sorted(
            os.path.join(ref_dir, f)
            for f in os.listdir(ref_dir)
            if f.endswith(".asm")
        )
        if not ref_paths:
            print(f"[SIM][SpecEnv] no .asm files in {ref_dir}; similarity reward disabled")
            return
        try:
            tok, model = _get_text_embedder(self.similarity_model_name, self.similarity_device)
        except Exception as exc:  # noqa: BLE001
            print(f"[SIM][SpecEnv] failed to load embedder {self.similarity_model_name}: {exc}; "
                  "similarity reward disabled")
            return

        ref_texts = [_filter_reference_asm_text(p) for p in ref_paths]
        empty = [p for p, t in zip(ref_paths, ref_texts) if not t.strip()]
        if empty:
            print(f"[SIM][SpecEnv] WARNING: empty after filtering: {empty}")

        self._sim_tokenizer = tok
        self._sim_model = model
        self._reference_embeddings = self._embed_texts(ref_texts)
        # Centroid (normalized) for optional baseline subtraction.
        import torch
        centroid = self._reference_embeddings.mean(dim=0, keepdim=True)
        self._reference_centroid = torch.nn.functional.normalize(centroid, dim=-1).squeeze(0)

        # Diagnostic: print pairwise cosines so the user can see how
        # discriminative the references are (high baseline => weak signal).
        sim = (self._reference_embeddings @ self._reference_embeddings.T).cpu().numpy()
        off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
        print(f"[SIM][SpecEnv] loaded {len(ref_paths)} refs from {ref_dir}; "
              f"pairwise cosine: mean={off_diag.mean():.3f}, "
              f"min={off_diag.min():.3f}, max={off_diag.max():.3f}")

    def _embed_texts(self, texts: List[str]):
        """Tokenize + forward + mean-pool + L2-normalize. Returns (B, hidden_dim)."""
        import torch
        with torch.no_grad():
            enc = self._sim_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.similarity_max_length,
                return_tensors="pt",
            ).to(self.similarity_device)
            out = self._sim_model(**enc).last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return torch.nn.functional.normalize(pooled, dim=-1)

    # SSB / SSBD helper methods. Originally added for Spectre v4 only; kept
    # always-on because Ray rollout workers re-import CONF and would otherwise
    # default x86_executor_enable_ssbp_patch=True, leaving SSBD MSR=1 in
    # ring-0 test runs and silently disabling v4 even if a v4-shaped gadget
    # is produced. Harmless under v1.
    def _read_ssb_status_line(self) -> str:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Speculation_Store_Bypass:"):
                        return line.strip()
        except OSError:
            return "Speculation_Store_Bypass: unknown (read failed)"
        return "Speculation_Store_Bypass: unknown (missing)"

    def _force_kernel_ssb_vulnerable(self) -> None:
        """
        Directly write "0" to /sys/rvzr_executor/enable_ssbp_patch so that the
        kernel module clears MSR_IA32_SPEC_CTRL.SSBD when running test cases
        in ring 0. Also force CONF.x86_executor_enable_ssbp_patch = False so
        that subsequent Executor operations (which re-sync CONF into sysfs)
        don't flip it back on.

        The sysfs file is write-only (EIO on read), so we can't verify after
        the write; we log whether the write succeeded and trust the kernel
        module's own error reporting.
        """
        # Force CONF flag so any later _set_vendor_specific_features() call
        # during re-init of the executor keeps SSBD cleared.
        try:
            setattr(CONF, "x86_executor_enable_ssbp_patch", False)
        except Exception as exc:  # noqa: BLE001
            print(f"[SSBD][SpecEnv] could not set CONF flag: {exc}")

        sysfs = "/sys/rvzr_executor/enable_ssbp_patch"
        try:
            with open(sysfs, "w", encoding="utf-8") as f:
                f.write("0")
            print(f"[SSBD][SpecEnv] wrote '0' to {sysfs} (SSB vulnerable in executor)")
        except OSError as exc:
            print(
                f"[SSBD][SpecEnv] FAILED to write {sysfs}: {exc}. "
                "Kernel module may not be loaded or permissions are wrong. "
                "Without this, Spectre v4 CANNOT be triggered."
            )

    def _ensure_ssb_vulnerable_for_process(self) -> None:
        """
        Ask kernel to enable speculative store bypass for this process/thread.
        Safe no-op if kernel policy refuses it.

        Also logs the system-wide SSB vulnerability status so that a failed v4
        training run can be attributed to CPU / microcode mitigations rather
        than to the RL policy.  If the machine reports the SSB channel as
        "not affected" or "Mitigation: Speculative Store Bypass disabled ...",
        no amount of reward shaping will produce a v4 leak.
        """
        libc = ctypes.CDLL(None, use_errno=True)
        before = self._read_ssb_status_line()
        rc = libc.prctl(
            ctypes.c_int(PR_SET_SPECULATION_CTRL),
            ctypes.c_ulong(PR_SPEC_STORE_BYPASS),
            ctypes.c_ulong(PR_SPEC_ENABLE),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
        if rc != 0:
            err = ctypes.get_errno()
            print(f"[SSB][SpecEnv] prctl(PR_SPEC_ENABLE) failed (errno={err})")
        after = self._read_ssb_status_line()
        print(f"[SSB][SpecEnv] {before} -> {after}")

        sysfs_path = "/sys/devices/system/cpu/vulnerabilities/spec_store_bypass"
        try:
            with open(sysfs_path, "r", encoding="utf-8") as f:
                sys_status = f.read().strip()
            print(f"[SSB][SpecEnv] sysfs {sysfs_path}: {sys_status}")
            lowered = sys_status.lower()
            # Only warn on genuinely-locked states. Strings like
            # "Mitigation: Speculative Store Bypass disabled via prctl"
            # historically tripped the warning because they contain
            # "disabled", but that's actually the per-process opt-out
            # mode that the kernel module + our _force_kernel_ssb_vulnerable
            # override anyway. Be explicit about which states really kill v4.
            hard_locked = (
                "not affected" in lowered
                or "mitigation: speculative store bypass disabled" in lowered
            )
            is_prctl_mode = "disabled via prctl" in lowered
            if hard_locked and not is_prctl_mode:
                print(
                    "[SSB][SpecEnv] WARNING: CPU/kernel reports SSB as "
                    "not exploitable (hard mitigation). Spectre v4 leaks "
                    "will not be detected regardless of agent policy or "
                    "kernel module flags. Check microcode / BIOS / "
                    "boot-time mitigations parameters."
                )
        except OSError as exc:
            print(f"[SSB][SpecEnv] could not read {sysfs_path}: {exc}")

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
        # action = (opcode, reg_src, reg_dst, imm)
        parts = (action[0], action[1], action[2], action[3]) if isinstance(action, (tuple, list)) else action
        o = int(parts[0].item() if hasattr(parts[0], "item") else parts[0])
        rs = int(parts[1].item() if hasattr(parts[1], "item") else parts[1])
        rd = int(parts[2].item() if hasattr(parts[2], "item") else parts[2])
        imm = int(parts[3].item() if hasattr(parts[3], "item") else parts[3])
        end = (o, rs, rd, imm) == (0, 0, 0, 0)

        # Block early-exit: the optimal policy under the old reward was to pick
        # end_game on step 0 to minimize the -81/step floor. With reward shaping
        # tightened, also suppress end_game until the agent has emitted enough
        # tokens to possibly form a v4 gadget. Treated as an invalid step.
        if end and self.num_steps < self.min_steps_before_end:
            print(
                f"EARLY END_GAME blocked (num_steps={self.num_steps} < "
                f"min_steps_before_end={self.min_steps_before_end})"
            )
            truncate = self.num_steps >= self.max_steps
            step_obs = self._get_obs()
            return step_obs, -self.early_end_penalty, False, truncate, {
                "program": self.new_program,
                "end_game_count": self.end_game_counter,
                "early_end_blocked": True,
            }

        inst_action = self._tuple_to_instruction(o, rs, rd, imm) if not end else None

        truncate = self.num_steps >= self.max_steps
        if end:
            self.end_game_counter += 1
            print(f"NUMBER OF END_GAME: {self.end_game_counter}")
        if not end and not truncate and inst_action is not None:

            self.step_counter += 1
            print(f"NUMBER OF STEPS: {self.step_counter}")

            target_desc = X86TargetDesc()
            self.generator._insert_bb_index = random.choice([0, 1])
            # self.generator._insert_bb_index = 0
            passed_inst = X86CheckAll(self.generator, self.new_program, inst_action, target_desc)
            # Only run the (expensive, assembles+emulates) loop check on instructions
            # that already passed validation — assembling an invalid instruction would
            # raise inside assemble() and crash the episode.
            passed_loop = passed_inst and self._infiniteLoopCheck(self.new_program, inst_action, 1)
            # passed_loop = True
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
        # Violation found -> terminate the episode immediately and move on to the
        # next test case, instead of appending more instructions onto an already
        # violating program. _reward() -> _full_obs_program() set self.leak.
        terminated_on_violation = bool(self.leak)
        if terminated_on_violation:
            print("VIOLATION FOUND -> terminating episode immediately, starting next round")
            end = True
        return step_obs, step_reward, end, truncate, {
            "program": self.new_program,
            "end_game_count": self.end_game_counter,
            "pattern_reward": self.pattern_stage_reward,
            "pattern_matches": self.pattern_match_counts,
            "terminated_on_violation": terminated_on_violation,
        }


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
        # Completed-episode snapshot for RLlib callbacks (works whether on_episode_end runs before or after reset).
        self._last_completed_episode_pattern_match = dict(self.episode_pattern_match_max)
        self.episode_pattern_match_max = {}
        # Same snapshot strategy for micro-arch signal roll-ups.
        self._last_episode_trace_div_max = float(self.episode_trace_div_max)
        self._last_episode_observable_any = bool(self.episode_observable_any)
        self._last_episode_leak_any = bool(self.episode_leak_any)
        self._last_episode_steps_observable = int(self.episode_steps_observable)
        self._last_episode_similarity_max = float(self.episode_similarity_max)
        self.episode_trace_div_max = 0.0
        self.episode_observable_any = False
        self.episode_leak_any = False
        self.episode_steps_observable = 0
        self.episode_similarity_max = 0.0
        self.num_steps = 0
        self.bad_case = False
        self.pattern_stage_reward = 0.0
        self.pattern_match_counts = {}
        self.trace_divergence_score = 0.0
        self.similarity_score = 0.0
        self.new_program = self.generator.create_test_case_SpecRL("/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/my_test_case.asm", disable_assembler=True, generate_empty_case=True)

        # Re-sample inputs every episode so the htrace side-channel signal
        # actually varies across episodes — v4 observability relies on the
        # stored value vs. pre-existing memory value differing, and a fixed
        # input set from __init__ collapses that variance.
        try:
            self.inputs = self.input_gen.generate(self.num_inputs, n_actors=1)
        except Exception as exc:
            print(f"[SpecEnv.reset] input regeneration failed ({exc}); reusing old inputs")

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
            "transient_uops": np.full((self.seq_size, self.num_inputs), -1, dtype=np.int64),
            "regs": np.full((self.seq_size, self.num_inputs, self.num_arch_regs), 0, dtype=np.uint8),
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
                os.makedirs("/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/debug_asm", exist_ok=True)
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

                # print(f"\niteration {count} observations: ")

                # fill appropriate observation row in
                # obs["instruction"][count - 1] = temp_obs[0]
                obs["instruction"][count - 1] = np.array(temp_obs[0])
                # htrace/ctrace: (num_inputs,) per step, stored in (seq_size, num_inputs) array
                temp_htrace = np.array(temp_obs[1], dtype=np.int64)
                obs["htrace"][count - 1, : temp_htrace.shape[0]] = temp_htrace[: self.num_inputs]

                # temp_ctrace = np.array(temp_obs[2], dtype=np.int64)
                temp_ctrace = np.array(temp_obs[2], dtype=np.uint64)
                obs["ctrace"][count - 1, : temp_ctrace.shape[0]] = temp_ctrace[: self.num_inputs]

                obs["recovery_cycles"][count - 1] = temp_obs[3]
                obs["transient_uops"][count - 1] = temp_obs[4]

                temp_regs = np.array(temp_obs[5], dtype=np.uint8)
                obs["regs"][count - 1, : temp_regs.shape[0]] = temp_regs[: self.num_inputs]

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
        # PFC counters come back as uint64 from the kernel module. In rare
        # configs the top bit can be set (counter wrap / uninitialized state /
        # error sentinel), and the resulting Python int exceeds int64.max, so
        # numpy's assignment into the int64 obs buffers throws
        # `OverflowError: Python int too large to convert to C long`.
        # The observation space already declares `high = int32.max` for both
        # recovery_cycles and transient_uops, so clip to that range here.
        _PFC_HIGH = int(np.iinfo(np.int32).max)
        recovery = []
        transient = []
        pfc_feedback = [ht.get_max_pfc() for ht in htraces]
        for _, pfc_values in enumerate(pfc_feedback):
            rec = int(pfc_values[2])
            if rec < 0:
                rec = -1
            elif rec > _PFC_HIGH:
                rec = _PFC_HIGH
            recovery.append(rec)
            if pfc_values[0] > pfc_values[1]:
                diff = int(pfc_values[0]) - int(pfc_values[1])
                if diff < 0:
                    diff = 0
                elif diff > _PFC_HIGH:
                    diff = _PFC_HIGH
                transient.append(diff)
            else:
                transient.append(0)
        last_instr_info = tuple(self._extract_last_instr_info(program))
        regs_obs = self._collect_reg_state(program)
        # print("regs_obs", regs_obs)
        return (last_instr_info, htraces_obs, ctraces_obs, recovery, transient, regs_obs)

    def _collect_reg_state(self, program: TestCaseProgram) -> List[List[int]]:
        """
        Trace the test case under enable_dbg_gpr_mode and return a list of
        per-input register-equality vectors (length=num_arch_regs, values 0/1).

        Encoding rule: for each slot in [rax, rbx, rcx, rdx, rsi, rdi],
        output 1 if its value duplicates at least one OTHER slot in the same
        snapshot, else 0. Raw absolute values are mostly sandbox_base+offset
        and carry little signal; what matters for v4 gadget structure is
        register aliasing (e.g., is the store-base also the load-dst?).

        Caveat: this encoding cannot tell {rax==rbx, rcx==rdx} apart from
        {rax==rbx==rcx==rdx} — both produce [1,1,1,1,0,0]. If that
        ambiguity hurts, switch to a pairwise encoding (15 bits over the
        C(6,2) reg pairs) which captures the full equivalence relation.

        n_reps=1 because the architectural state is deterministic.
        """
        self.arch_executor.load_test_case(program)
        reg_traces = self.arch_executor.trace_test_case(self.inputs, n_reps=1)
        regs: List[List[int]] = []
        for rt in reg_traces:
            samples = rt.get_raw_readings()
            if len(samples) == 0:
                regs.append([0] * self.num_arch_regs)
                continue
            s = samples[0]
            raw = [
                int(s['trace']),
                int(s['pfc0']),
                int(s['pfc1']),
                int(s['pfc2']),
                int(s['pfc3']),
                int(s['pfc4']),
            ]
            counts = Counter(raw)
            regs.append([1 if counts[v] > 1 else 0 for v in raw])
        return regs

    def _extract_agent_asm_text(self, program: TestCaseProgram) -> str:
        """
        Render the agent-emitted instructions of `program` as plain asm text,
        one instruction per line. Skips the same things _build_pattern_tokens
        and _filter_reference_asm_text skip:
            - instrumentation (sandbox-inserted ops)
            - macros (.macro.measurement_*)
        Used as the input text to the similarity embedder.
        """
        lines: List[str] = []
        for bb in program.iter_basic_blocks():
            for instr in bb:
                if instr.is_instrumentation:
                    continue
                if instr.name == "macro":
                    continue
                lines.append(self.printer._instruction_to_str(instr))
        return "\n".join(lines)

    def _compute_similarity_score(self, program: TestCaseProgram) -> float:
        """
        Cosine similarity (top-K mean) between embedding of agent's asm text
        and the cached reference embeddings. Returns 0.0 if similarity reward
        is disabled or program has no agent-emitted instructions.
        """
        if self._reference_embeddings is None or self._sim_model is None:
            return 0.0
        text = self._extract_agent_asm_text(program)
        if not text.strip():
            return 0.0
        agent_emb = self._embed_texts([text])  # (1, H), normalized
        sims = (agent_emb @ self._reference_embeddings.T).squeeze(0)  # (N_ref,)

        if self.similarity_baseline_subtract:
            # Subtract similarity to the centroid: rewards being closer to
            # the references than the references' average is to itself.
            baseline = float((agent_emb @ self._reference_centroid).item())
            sims = sims - baseline

        k = min(self.similarity_top_k, sims.shape[0])
        topk = sims.topk(k).values
        return float(topk.mean().item())


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
        # reward += self.pattern_stage_reward
        reward += self.trace_divergence_reward_scale * self.trace_divergence_score
        # Code-similarity reward (UniXcoder cosine vs reference attacks).
        # 0 when disabled (no refs / model failed to load) so it's a no-op.
        # sim_reward = self.similarity_reward_scale * self.similarity_score
        # reward += sim_reward
        # for k, v in self.pattern_match_counts.items():
        #     iv = int(v)
        #     self.episode_pattern_match_max[k] = max(self.episode_pattern_match_max.get(k, 0), iv)
        # Per-episode micro-arch signal roll-ups (feed wandb via callbacks).
        # if self.trace_divergence_score > self.episode_trace_div_max:
        #     self.episode_trace_div_max = float(self.trace_divergence_score)
        # if self.similarity_score > self.episode_similarity_max:
        #     self.episode_similarity_max = float(self.similarity_score)
        if self.observable:
            self.episode_observable_any = True
            self.episode_steps_observable += 1
        if self.leak:
            self.episode_leak_any = True
        # print(f"pattern shaping reward: {self.pattern_stage_reward}, matches: {self.pattern_match_counts}")
        print(
            f"trace divergence reward: {self.trace_divergence_reward_scale * self.trace_divergence_score} "
            f"(score={self.trace_divergence_score})"
        )
        # print(f"similarity reward: {sim_reward} (score={self.similarity_score:.4f}, top_k={self.similarity_top_k})")
        if self.leak:
            print("\n!!! LEAK OCCURED !!!\n")
            reward += self.leak_reward
        # Misspec filter: ±self.misspec_bonus. Defaults to 0 for spectre_v4
        # (PFC branch/machine-clear counters don't fire for SSB speculation).
        if self.misspec_bonus != 0.0:
            reward = reward + self.misspec_bonus if self.misspec else reward - self.misspec_bonus
        print(f"misspec: {self.misspec}, reward: {reward}")
        # Observable filter: single-sided (P0.2). +observable_bonus when the
        # fenced-vs-unfenced htrace diverges (a real pre-leak signal), and
        # -observable_penalty (default 0) otherwise so non-observable steps
        # don't accumulate a huge negative drag against pattern shaping.
        if self.observable:
            reward += self.observable_bonus
        else:
            reward -= self.observable_penalty
        print(f"observable: {self.observable}, reward: {reward}")
        reward -= self.step_penalty
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
        self.pattern_stage_reward = 0.0
        self.pattern_match_counts = {}
        self.trace_divergence_score = 0.0
        self.similarity_score = 0.0

        self.fuzzer = X86Fuzzer("/home/mluo/sca-fuzzer/base.json", os.getcwd(), existing_test_case= "/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/my_test_case.asm", input_paths=self.inputs)
        self.fuzzer.model = self.model
        self.fuzzer.data_gen = self.input_gen
        self.fuzzer.analyser = self.analyser
        self.fuzzer.executor = self.executor
        asm = tempfile.NamedTemporaryFile(delete=False)
        bin = tempfile.NamedTemporaryFile(delete=False)
        fenced = None
        fenced_obj = None

        def _safe_remove(path: Optional[str]) -> None:
            if path and os.path.exists(path):
                os.remove(path)

        try:
            temp = copy.deepcopy(program)

            all_instructions = []
            for bb in temp.iter_basic_blocks():
                for instr in list(bb):
                    instr._section_id = -1
                    all_instructions.append(instr)
            total_instructions_num = len(all_instructions)
            print("total instructions in program: ", total_instructions_num)
            print("all instructions: ", [self.printer._instruction_to_str(instr) for instr in all_instructions])
            # pattern_tokens = self._build_pattern_tokens(all_instructions)
            # pattern_result = self.pattern_matcher.score(pattern_tokens)
            # self.pattern_stage_reward = self.pattern_reward_scale * pattern_result.score
            # self.pattern_match_counts = pattern_result.matches

            # Code-similarity (UniXcoder cosine vs reference attacks). Cheap
            # to compute alongside pattern matching since `temp` already has
            # the agent-emitted instructions in iteration order.
            # self.similarity_score = self._compute_similarity_score(temp)

            temp.assign_obj(bin.name)
            assemble(temp)
            self.elf_parser.populate_elf_data(temp.get_obj(), temp)

            manager = _RoundManager(self.fuzzer, temp, inputs)
            manager._boost_inputs()
            boosted_inputs = manager.boosted_inputs

            # check for violations
            ctraces = self.model.trace_test_case(boosted_inputs, 1)
            # P2.2: more reps -> more stable fenced/unfenced comparison.
            # _get_obs may have left /sys/rvzr_executor/enable_dbg_gpr_mode=1
            # via arch_executor. Re-load through self.executor to flip the
            # global sysfs flag back to 0 so trace_test_case returns cache
            # htraces, not GPR snapshots.
            self.executor.load_test_case(temp)
            htraces = self.executor.trace_test_case(boosted_inputs, 20)

            # check if misspec occurs, updates flag
            pfc_feedback = [ht.get_max_pfc() for ht in htraces]
            for i, pfc_values in enumerate(pfc_feedback):
                if pfc_values[0] > pfc_values[1] or pfc_values[2] > 0:
                    self.misspec = True

            # check if it's observable, updates flag
            fenced = tempfile.NamedTemporaryFile(delete=False)
            fenced_obj = tempfile.NamedTemporaryFile(delete=False)

            debug_path = "/home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing/debug_asm"
            os.makedirs(debug_path, exist_ok=True)
            fenced_test_case = _create_fenced_test_case(temp._asm_path, fenced.name, self.asm_parser, self.generator, self.elf_parser)

            self.executor.load_test_case(fenced_test_case)
            fenced_htraces = self.executor.trace_test_case(inputs, n_reps=20)

            if fenced_htraces != htraces:
                self.observable = True

            # Dense pre-leak signal: normalized count of input traces that diverge
            # between fenced and unfenced runs.
            div_count = 0
            max_n = min(len(inputs), len(htraces), len(fenced_htraces))
            for idx in range(max_n):
                if not self.analyser.htraces_are_equivalent(fenced_htraces[idx], htraces[idx]):
                    div_count += 1
            self.trace_divergence_score = float(div_count) / float(max_n or 1)

            # traces_match = True
            # for i, _ in enumerate(inputs):
            #     if not self.analyser.htraces_are_equivalent(fenced_htraces[i], htraces[i]):
            #         traces_match = False
            #         break
            # print("traces_match ***************", traces_match)

            violations = self.fuzzer.start_SpecRL(1, len(inputs), 0, False, True, type_='asm')
            if not violations:  # nothing detected? -> we are done here, move to next test case
                return None

            print("\n\n\nFOUND VIOLATION!!!\n\n\n")
            self._save_violation_asm(temp, debug_path)

            # 2. Repeat with with max nesting
            # if 'seq' not in CONF.contract_execution_clause:
                # self.LOG.fuzzer_nesting_increased()
                #boosted_inputs = fuzzer.boost_inputs(inputs, CONF.model_max_nesting)
                # manager = _RoundManager(self.fuzzer, temp, inputs)
                # manager._boost_inputs()
                # boosted_inputs = manager.boosted_inputs
                # ctraces = self.model.trace_test_case(boosted_inputs, CONF.model_max_nesting)
                # htraces = self.executor.trace_test_case(boosted_inputs, CONF.executor_repetitions)
                # violations = self.fuzzer.analyser.filter_violations(boosted_inputs, ctraces, htraces, True)
                # if not violations:
                #     print("\n\n\n FAILED MAX NESTING \n\n\n")
                #     return None

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
        finally:
            _safe_remove(fenced.name if fenced is not None else None)
            _safe_remove(fenced_obj.name if fenced_obj is not None else None)
            _safe_remove(asm.name)
            _safe_remove(bin.name)

    def _save_violation_asm(self, program: TestCaseProgram, debug_path: str) -> Optional[str]:
        """
        Persist the current test case as ASM when a violation is detected.
        Writes to debug_path/violation_<timestamp>.asm
        """
        os.makedirs(debug_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(debug_path, f"violation_{ts}.asm")
        p = copy.deepcopy(program)
        p._asm_path = out_path
        self.printer.print(p)
        print(f"[SpecEnv] Saved violation ASM -> {out_path}")
        return out_path

    def _infiniteLoopCheck(self, prog: TestCaseProgram, instr: Instruction, timeout: int) -> bool:
        prog_ = copy.deepcopy(prog)
        unique = uuid.uuid4().hex
        temp_asm_path = f"/tmp/test_case_{unique}.s"
        temp_bin_path = f"/tmp/test_case_{unique}.o"
        # Point the throwaway program at its own temp files BEFORE inserting: the
        # insert routine prints the program internally, and we don't want it writing
        # the candidate (instruction-added) asm over the shared source path.
        prog_._asm_path = temp_asm_path
        prog_.assign_obj(temp_bin_path)
        try:
            # Insert using the same path/bb index as the real insertion in step(), so the
            # loop check validates the exact program layout that will be constructed.
            self.generator.insert_instruction_in_test_case_randomly(
                prog_, instr, self.generator._insert_bb_index)
            self.printer.print(prog_)
            assemble(prog_)
            self.elf_parser.populate_elf_data(prog_.get_obj(), prog_)
            self.model.load_test_case(prog_)
            return self.model.check_inf_loop(self.inputs, 1, timeout)
        except CalledProcessError:
            # Failed to assemble — treat as not passing so the step is thrown away
            # rather than crashing the episode.
            return False
        finally:
            for path in (temp_asm_path, temp_bin_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

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

    def _get_instruction_variant_name(self, instr: Instruction) -> str:
        """Map a concrete instruction back to the action-space opcode variant."""
        name_lower = instr.name.lower()

        # Variants we emit via inst_space.tuple_to_instruction have canonical shapes,
        # so map them back directly to the opcode vocab used in OPERAND_SPACE.
        if name_lower == "lea":
            return "lea_rrr"
        if name_lower == "xor":
            return "xor_rr"
        if name_lower == "mul":
            return "mul"

        # New pure-GPR opcodes — name is unique, no operand-shape ambiguity.
        if name_lower == "div":
            return "div_r"
        if name_lower == "idiv":
            return "idiv_r"
        if name_lower == "lock xadd":
            return "lock_xadd_mr"
        if name_lower == "xchg":
            return "lock_xchg_mr"
        if name_lower == "lock cmpxchg":
            return "lock_cmpxchg_mr"
        if name_lower == "lock add":
            return "lock_add_mr"
        if name_lower == "repe movsb":
            return "rep_movsb"
        if name_lower == "repe stosb":
            return "rep_stosb"
        if name_lower == "repe cmpsb":
            return "rep_cmpsb"
        if name_lower == "repe scasb":
            return "rep_scasb"
        # BASE-CMOV (only _rm form is exposed). cmova/cmovnbe collapse to one entry.
        if name_lower == "cmovz":
            return "cmovz_rm"
        if name_lower == "cmovnz":
            return "cmovnz_rm"
        if name_lower == "cmovb":
            return "cmovb_rm"
        if name_lower in ("cmova", "cmovnbe"):
            return "cmova_rm"
        # sbb has only the _rr variant in OPERAND_SPACE; no shape disambiguation needed.
        if name_lower == "sbb":
            return "sbb_rr"
        # lock not is the single-operand mem RMW.
        if name_lower == "lock not":
            return "lock_not_m"

        # XMM (SSE / SSE2) variants — name alone or shape disambiguates.
        if name_lower in ("pxor", "paddq", "pmuludq"):
            return name_lower + "_xx"
        if name_lower == "movdqu":
            # _xm = XMM load (REG128, MEM128); _mx = store (MEM128, REG128)
            return "movdqu_xm" if len(instr.get_mem_operands()) > 0 and any(
                op.dest for op in instr.get_reg_operands()
            ) else "movdqu_mx"
        if name_lower == "movups":
            return "movups_xm" if len(instr.get_mem_operands()) > 0 and any(
                op.dest for op in instr.get_reg_operands()
            ) else "movups_mx"
        if name_lower == "movq":
            # Only movq_xr is exposed in OPERAND_SPACE (GPR -> XMM, REG128,REG64).
            return "movq_xr"
        if name_lower == "movd":
            # Only movd_xr (GPR -> XMM, REG128,REG32) remains in OPERAND_SPACE;
            # movd_rx (XMM -> GPR) was removed. A `movd xmm, reg32` (dst REG128)
            # maps to movd_xr; the old `movd reg32, xmm` shape now has no action
            # and stays unmapped ("movd" is not in opcode_vocab -> caller id -1).
            for op in instr.get_reg_operands():
                if op.dest and op.width >= 128:
                    return "movd_xr"
            return "movd"  # unmapped

        if name_lower not in {"mov", "add", "cmp", "sub"}:
            return name_lower

        has_mem = len(instr.get_mem_operands()) > 0
        has_imm = len(instr.get_imm_operands()) > 0
        reg_ops = instr.get_reg_operands(include_implicit=False)
        has_reg_src = any(op.src for op in reg_ops)
        has_reg_dst = any(op.dest for op in reg_ops)

        if name_lower == "mov":
            if has_reg_dst and has_reg_src and not has_mem and not has_imm:
                return "mov_rr"
            if has_reg_dst and has_imm and not has_mem:
                return "mov_ri"
            if has_reg_dst and has_mem:
                return "mov_rm"
            if has_mem and has_reg_src:
                return "mov_mr"
            if has_mem and has_imm:
                return "mov_mi"

        if name_lower == "add":
            if has_reg_dst and has_reg_src and not has_mem and not has_imm:
                return "add_rr"
            if has_reg_dst and has_imm and not has_mem:
                return "add_ri"
            if has_reg_dst and has_mem:
                return "add_rm"
            if has_mem and has_reg_src:
                return "add_mr"
            if has_mem and has_imm:
                return "add_mi"

        if name_lower == "cmp":
            if has_reg_dst and has_reg_src and not has_mem and not has_imm:
                return "cmp_rr"
            if has_reg_dst and has_mem:
                return "cmp_rm"
            if has_mem and has_reg_src:
                return "cmp_mr"

        if name_lower == "sub":
            if has_reg_dst and has_reg_src and not has_mem and not has_imm:
                return "sub_rr"
            if has_reg_dst and has_imm and not has_mem:
                return "sub_ri"
            if has_reg_dst and has_mem:
                return "sub_rm"
            if has_mem and has_reg_src:
                return "sub_mr"
            if has_mem and has_imm:
                return "sub_mi"

        return name_lower
    
    def _extract_last_instr_info(self, program: TestCaseProgram) -> List[int]:
        """Return [opname_id, reg_src_id, reg_dst_id, imm_id] for last instruction."""
        last_instr = self._get_last_instruction(program)
        if last_instr is None:
            return [-1, -1, -1, -1]

        variant_name = self._get_instruction_variant_name(last_instr)
        opname_id = self.opcode_vocab.index(variant_name) if variant_name in self.opcode_vocab else -1

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

    def _build_pattern_tokens(self, instructions: List[Instruction]) -> List[PatternToken]:
        tokens: List[PatternToken] = []
        for instr in instructions:
            if instr.is_instrumentation or instr.name == "macro":
                continue

            variant = self._get_instruction_variant_name(instr)
            mem_reads, mem_writes = self._extract_memory_access_bases(instr)
            dst_regs, src_regs = self._extract_gpr_side_effects(instr)
            kinds = {variant, variant.split("_")[0]}
            if mem_reads:
                kinds.add("load")
            if mem_writes:
                kinds.add("store")
            if mem_reads or mem_writes:
                kinds.add("memory")

            tokens.append(
                PatternToken(
                    index=len(tokens),
                    kinds=frozenset(kinds),
                    mem_reads=tuple(mem_reads),
                    mem_writes=tuple(mem_writes),
                    opcode_variant=variant,
                    dst_regs=tuple(dst_regs),
                    src_regs=tuple(src_regs),
                )
            )
        return tokens

    def _extract_gpr_side_effects(self, instr: Instruction) -> Tuple[List[str], List[str]]:
        """
        Return (dst_regs, src_regs) for the instruction, restricted to the env's reg_vocab.
        These are GPR-level side effects (ALU dst / load dst / ALU src), NOT memory-base
        registers used to compute addresses.  Memory bases are already handled by
        _extract_memory_access_bases.

        This powers dependency-chain reasoning in pattern_matching.py:
          - "is the store's base overwritten between store and load?"  -> dst_regs
          - "is there a slow-addr producer that WRITES the store's base?"  -> dst_regs
          - "is the transmitter's base a register written by the bypass load?" -> dst_regs
        """
        dst_regs: List[str] = []
        src_regs: List[str] = []
        for reg_op in instr.get_reg_operands(include_implicit=True):
            name = self._normalize_reg_name(str(reg_op.value).lower())
            if name not in self.reg_vocab:
                continue
            if getattr(reg_op, "dest", False):
                dst_regs.append(name)
            if getattr(reg_op, "src", False):
                src_regs.append(name)

        # Annotate implicit register side effects that aren't surfaced as
        # RegisterOps in our synthesized instructions. This matters for
        # pattern_matching._has_slow_addr_producer to credit `mul` as a
        # producer of the store's base register (rax or rdx).
        name_lower = instr.name.lower()
        if name_lower == "mul":
            # mul writes rdx:rax implicitly, reads rax implicitly.
            dst_regs.extend(["rax", "rdx"])
            src_regs.append("rax")

        return sorted(set(dst_regs)), sorted(set(src_regs))

    def _extract_memory_access_bases(self, instr: Instruction) -> tuple[List[str], List[str]]:
        reads: List[str] = []
        writes: List[str] = []
        for mem_op in instr.get_mem_operands(include_explicit=True):
            expr = str(mem_op.value).lower()
            parts = re.split(r"[^a-z0-9_]+", expr)
            bases = []
            for token in parts:
                if not token:
                    continue
                normalized = self._normalize_reg_name(token)
                if normalized in self.reg_vocab:
                    bases.append(normalized)
            if not bases:
                continue

            if getattr(mem_op, "src", False):
                reads.extend(bases)
            if getattr(mem_op, "dest", False):
                writes.extend(bases)
            if not getattr(mem_op, "src", False) and not getattr(mem_op, "dest", False):
                reads.extend(bases)

        return sorted(set(reads)), sorted(set(writes))
