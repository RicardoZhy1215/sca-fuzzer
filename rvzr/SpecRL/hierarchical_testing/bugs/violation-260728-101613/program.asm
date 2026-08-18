.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rax
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
por xmm0, xmm7
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
cmp rcx, rax
dec rcx
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi]
pextrq rbx, xmm4, 0
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4280
and rbx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax]
not rcx
mov rdi, 4520
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rsi]
por xmm4, xmm4
neg rcx
por xmm0, xmm7
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
