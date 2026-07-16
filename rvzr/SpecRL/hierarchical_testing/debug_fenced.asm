.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 45 # instrumentation
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
jnb .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
