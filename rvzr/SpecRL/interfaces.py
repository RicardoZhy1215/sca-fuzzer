
from rvzr.traces import CTrace
from rvzr.traces import HTrace
from ..tc_components.test_case_data import InputData
from typing import List, Dict, Tuple, Optional, NamedTuple
from collections import defaultdict


InputID = int

class Measurement(NamedTuple):
    input_id: InputID
    input_: InputData
    ctrace: CTrace
    htrace: HTrace


HTraceGroup = List[Measurement]
HTraceMap = Dict[HTrace, HTraceGroup]


class EquivalenceClass:
    ctrace: CTrace
    measurements: List[Measurement]
    htrace_map: HTraceMap
    MOD2P64 = pow(2, 64)

    def __init__(self) -> None:
        self.measurements = []

    def __str__(self):
        s = f"Size: {len(self.measurements)}\n"
        s += f"Ctrace:\n" \
             f"{self.ctrace % self.MOD2P64:064b} [ns]\n" \
             f"{(self.ctrace >> 64) % self.MOD2P64:064b} [s]\n"
        s += "Htraces:\n"
        for h in self.htrace_map.keys():
            s += f"{h:064b}\n"
        s = s.replace("0", "_").replace("1", "^")
        return s

    def __len__(self):
        return len(self.measurements)

    def build_htrace_map(self) -> None:
        """ group inputs by htraces """
        groups = defaultdict(list)
        for measurement in self.measurements:
            groups[measurement.htrace].append(measurement)
        self.htrace_map = groups