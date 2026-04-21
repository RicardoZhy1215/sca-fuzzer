.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
nop dword ptr [rax]  # 3 b
or ax, 1 # instrumentation
and dx, ax # instrumentation
shr dx, 1 # instrumentation
div ax
add bl, -77 # instrumentation
adc al, 20
nop dword ptr [rax + 0xff]  # 7 b
and rax, 0b1111111111111 # instrumentation
mul word ptr [r14 + rax]
nop qword ptr [rax + 1]  # 5 b
imul ax, ax
nop qword ptr [rax + rax + 1]  # 6 b
nop qword ptr [rax + rax + 1]  # 6 b
nop qword ptr [rax]  # 4 b
nop dword ptr [rax + 0xff]  # 7 b
add al, -5
jmp .exit_0

.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
