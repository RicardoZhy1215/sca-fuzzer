.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setb dil
pcmpeqd xmm4, xmm1
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rsi
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx
pmuludq xmm2, xmm0
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
pand xmm3, xmm4
inc rdx
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx]
setl al
dec rcx
pmuludq xmm0, xmm2
pmuludq xmm2, xmm6
imul rdx, rbx
jmp .bb_0.1
.bb_0.1:
not rsi
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
and rdi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdi]
pmuludq xmm6, xmm2
not rax
cmp rax, rbx
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rcx]
xor rbx, rsi
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rbx]
pextrq rdi, xmm0, 0
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rsi
psubq xmm3, xmm5
adc rcx, rax
and rdi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
