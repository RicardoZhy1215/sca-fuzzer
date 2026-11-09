# Unmodified Revizor on configs/v1.yaml: 3000 programs, -i 50

The run was cut short by an unclean machine restart at 1824 programs (see ABORTED.txt). The
remaining programs ran as `revizor-v1yaml-3k-part2-n1176-i50-261109-2156` with the same config
copy. Part 1 ran on kernel 6.8.0-139, part 2 on 6.8.0-147, both with SMT off and CPUs 0-7 online.
Joined curve: `../revizor-v1yaml-3k-JOINED-curve.txt` (from `revizor_curve.py <part1> <part2>`).

| part | programs | violations | per 1000 |
|------|----------|------------|----------|
| 1    | 1824     | 148        | 81.1     |
| 2    | 1175*    | 94         | 80.0     |
| all  | 2999*    | 242        | 80.7     |

\* As counted by Revizor's progress lines; part 2 was run with `-n 1176`.

Outcome of the programs: 1779 stopped at the speculation filter (59%), 433 at the observation
filter (14%), 545 at the fast path (18%), and 242 violated (8%). The cumulative curve is linear
(21 at 250, 78 at 1000, 163 at 2000, 242 at 2999). The two parts have the same rate.
