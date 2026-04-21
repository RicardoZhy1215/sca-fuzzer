.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
nop dword ptr [rax]  # 3 b
lfence
or ax, 1 # instrumentation
lfence
and dx, ax # instrumentation
lfence
shr dx, 1 # instrumentation
lfence
div ax
lfence
add bl, -77 # instrumentation
lfence
adc al, 20
lfence
nop dword ptr [rax + 0xff]  # 7 b
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul word ptr [r14 + rax]
lfence
nop qword ptr [rax + 1]  # 5 b
lfence
imul ax, ax
lfence
nop qword ptr [rax + rax + 1]  # 6 b
lfence
nop qword ptr [rax + rax + 1]  # 6 b
lfence
nop qword ptr [rax]  # 4 b
lfence
nop dword ptr [rax + 0xff]  # 7 b
lfence
add al, -5
lfence
jmp .exit_0

.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
