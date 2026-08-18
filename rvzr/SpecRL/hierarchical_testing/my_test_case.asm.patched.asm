.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3280
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rdi
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdx
and rax, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rax]
mov rdx, 1656
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
pmuludq xmm5, xmm5
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rcx
xor rdx, rbx
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
imul rdx, rdi
psubq xmm7, xmm2
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
