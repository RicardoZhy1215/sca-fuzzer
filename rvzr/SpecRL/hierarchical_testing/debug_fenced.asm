.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rax], 3280
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rdx]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rax], rdi
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rax]
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rdx], rcx
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rbx], rdx
lfence
and rax, 0b1111111111111 # instrumentation
lfence
cmp rdi, rdi # instrumentation
lfence
cmovz rdi, qword ptr [r14 + rax]
lfence
mov rdx, 1656
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdx], rdi
lfence
pmuludq xmm5, xmm5
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rsi], rcx
lfence
xor rdx, rbx
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rbx]
lfence
imul rdx, rdi
lfence
psubq xmm7, xmm2
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
