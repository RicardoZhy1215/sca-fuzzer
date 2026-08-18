.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pmuludq xmm2, xmm4
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rax]
cmp rax, rcx
pextrq rdi, xmm1, 0
pcmpeqd xmm6, xmm4
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax]
not rdi
lea rdx, qword ptr [rdx + rdx + 1]
setb bl
imul rbx, rax
setb dil
and rax, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rcx]
setb dil
por xmm1, xmm7
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rax
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
sub rdx, rax
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdi
pextrq rbx, xmm5, 0
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax
dec rdi
paddq xmm6, xmm3
adc rdi, rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
