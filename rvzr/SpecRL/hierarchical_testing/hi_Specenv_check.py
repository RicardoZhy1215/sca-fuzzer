import argparse
import ctypes
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SPECRL_ROOT = THIS_DIR.parent
RVZR_ROOT = SPECRL_ROOT.parent
REPO_ROOT = RVZR_ROOT.parent

# Match the import bootstrap used by hierarchical_testing/train.py so the script
# can be run from either the repo root or this directory.
for path in (str(REPO_ROOT), str(SPECRL_ROOT), str(THIS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rvzr.arch.x86.fuzzer import X86Fuzzer
from rvzr.config import CONF


PR_SET_SPECULATION_CTRL = 53
PR_SPEC_STORE_BYPASS = 0
PR_SPEC_ENABLE = 1 << 1

DEFAULT_CONFIG = THIS_DIR / "config.yaml"
DEFAULT_INCLUDE_DIR = THIS_DIR
DEFAULT_BASE_JSON = REPO_ROOT / "base.json"
DEFAULT_ASM = REPO_ROOT / "tests/x86_tests/asm/spectre_v4.asm"
DEFAULT_WORK_DIR = THIS_DIR


def _read_ssb_status_line() -> str:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Speculation_Store_Bypass:"):
                    return line.strip()
    except OSError:
        return "Speculation_Store_Bypass: unknown (read failed)"
    return "Speculation_Store_Bypass: unknown (missing)"


def ensure_ssb_vulnerable_for_process() -> None:
    """
    Request that the current process enables speculative store bypass when the
    kernel policy allows it. This mirrors the setup used by hi_SpecEnv.py.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    before = _read_ssb_status_line()

    rc = libc.prctl(
        ctypes.c_int(PR_SET_SPECULATION_CTRL),
        ctypes.c_ulong(PR_SPEC_STORE_BYPASS),
        ctypes.c_ulong(PR_SPEC_ENABLE),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if rc != 0:
        err = ctypes.get_errno()
        print(f"[SSB][hi_Specenv_check] prctl(PR_SPEC_ENABLE) failed (errno={err})")

    after = _read_ssb_status_line()
    print(f"[SSB][hi_Specenv_check] {before} -> {after}")

    sysfs_path = "/sys/devices/system/cpu/vulnerabilities/spec_store_bypass"
    try:
        with open(sysfs_path, "r", encoding="utf-8") as handle:
            sys_status = handle.read().strip()
        print(f"[SSB][hi_Specenv_check] sysfs {sysfs_path}: {sys_status}")
    except OSError as exc:
        print(f"[SSB][hi_Specenv_check] could not read {sysfs_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load spectre_v4.asm as-is and check whether it can trigger a "
            "violation in the SpecEnv / Revizor environment."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "-I",
        "--include-dir",
        default=str(DEFAULT_INCLUDE_DIR),
        help="Directory used for !include resolution in the config file.",
    )
    parser.add_argument(
        "--asm",
        default=str(DEFAULT_ASM),
        help="Path to the target assembly file. Default: spectre_v4.asm",
    )
    parser.add_argument(
        "--base-json",
        default=str(DEFAULT_BASE_JSON),
        help="Path to the ISA spec JSON.",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="Working directory for Revizor artifacts.",
    )
    parser.add_argument(
        "-i",
        "--num-inputs",
        type=int,
        default=100,
        help="Number of inputs to generate for the check.",
    )
    parser.add_argument(
        "-n",
        "--num-test-cases",
        type=int,
        default=1,
        help="Number of times to run the check.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout in seconds. 0 means no timeout.",
    )
    parser.add_argument(
        "--nonstop",
        action="store_true",
        help="Continue after the first detected violation.",
    )
    parser.add_argument(
        "--save-violations",
        action="store_true",
        help="Store violation artifacts in the working directory.",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    for label, path_str in (
        ("config", args.config),
        ("include_dir", args.include_dir),
        ("asm", args.asm),
        ("base_json", args.base_json),
        ("work_dir", args.work_dir),
    ):
        path = Path(path_str)
        if label in {"include_dir", "work_dir"}:
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"{label} is not a directory: {path}")
        else:
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"{label} is not a file: {path}")


def run_check(args: argparse.Namespace) -> bool:
    CONF.load(args.config, args.include_dir)
    ensure_ssb_vulnerable_for_process()

    # Key point: existing_test_case points directly to spectre_v4.asm, and we
    # use type_='asm', so Revizor parses and runs that file directly instead of
    # creating an empty testcase and inserting any additional instructions.
    fuzzer = X86Fuzzer(
        args.base_json,
        args.work_dir,
        existing_test_case=args.asm,
    )
    return fuzzer.start_SpecRL(
        args.num_test_cases,
        args.num_inputs,
        args.timeout,
        args.nonstop,
        args.save_violations,
        type_="asm",
    )


def main() -> int:
    args = parse_args()
    validate_paths(args)

    print(f"[hi_Specenv_check] config   : {args.config}")
    print(f"[hi_Specenv_check] asm      : {args.asm}")
    print(f"[hi_Specenv_check] base_json: {args.base_json}")
    print(f"[hi_Specenv_check] work_dir : {args.work_dir}")

    try:
        found_violation = run_check(args)
    except subprocess.CalledProcessError as exc:
        print("[hi_Specenv_check] Executor setup failed before the asm check could run.")
        print(
            "[hi_Specenv_check] Revizor likely could not write /sys/rvzr_executor/* "
            "or initialize the kernel module in this environment."
        )
        print(f"[hi_Specenv_check] Failing command: {exc.cmd}")
        return 2

    if found_violation:
        print("[hi_Specenv_check] Result: violation detected.")
        return 1

    print("[hi_Specenv_check] Result: no violation detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
