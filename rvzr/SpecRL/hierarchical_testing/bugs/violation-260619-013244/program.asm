.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
dec rcx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
cmp rsi, rcx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdi
por xmm3, xmm7
dec rsi
pcmpeqd xmm7, xmm5
imul rdx, rcx
xor rbx, rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdi
mov rdx, rsi
inc rcx
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
pextrq rsi, xmm4, 0
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
pcmpeqd xmm7, xmm3
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
