.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 15 # instrumentation
imul rdi, rdx
imul rdi, rdx
imul rdi, rdx
imul rdi, rdx
imul rdi, rdx
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
imul rdi, rdx
imul rdi, rdx
imul rdi, rdx
imul rdi, rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
