.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pmuludq xmm1, xmm3
and rdi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx]
or rdx, rcx
inc rbx
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rdi, rcx
setnz cl
and rax, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rax]
or rax, rcx
pextrq rsi, xmm2, 0
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1856
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
setl dil
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4752
and rdx, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
sbb rdi, rsi
and rcx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rcx]
jmp .bb_0.1
.bb_0.1:
pmuludq xmm2, xmm3
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
lea rdx, qword ptr [rsi + rdx + 1]
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx]
paddq xmm1, xmm7
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
imul rcx, rdi
paddq xmm0, xmm6
dec rcx
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rax
imul rcx, rdx
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx]
or rdx, rdi
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
psubq xmm0, xmm3
setb sil
and rsi, rbx
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rsi
pcmpeqd xmm2, xmm7
and rdx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdx]
movq xmm3, rbx
neg rax
and rsi, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rsi]
movq xmm1, rdi
sub rdi, rsi
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
