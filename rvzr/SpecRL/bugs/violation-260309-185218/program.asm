.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -89 # instrumentation
cmp rax, rbx
xor rdx, rbx
xor rdx, rbx
and rbx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rbx]
loope .bb_0.1
jmp .exit_0
.bb_0.1:
xor rdx, rbx
xor rdx, rbx
cmp rbx, rdx
cmp rbx, rdx
and rbx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rbx]
cmp rbx, rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
