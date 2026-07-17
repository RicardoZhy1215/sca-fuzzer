.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -50 # instrumentation
imul rdi, rdx
imul rdi, rdx
mov rdi, 6864
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
imul rdi, rdx
jp .bb_0.1
jmp .exit_0
.bb_0.1:
imul rdi, rdx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx
imul rdi, rdx
imul rdi, rdx
imul rdi, rsi
imul rdi, rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
