.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 123 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rdx
imul rsi, rdi
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
dec rdi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
or rdi, rcx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6968
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
setz dil
or rdi, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
