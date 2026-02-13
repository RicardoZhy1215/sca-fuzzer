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
jbe .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111000 # instrumentation
or word ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 #
and rdi, 0b1111111111000 # instrumentation
mov cl, byte ptr [r14 + rdi]
nop dword ptr [rax + 0xff]  # 7 b
nop dword ptr [rax + 0xff]  # 7 b
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
