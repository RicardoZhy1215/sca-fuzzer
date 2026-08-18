.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rdx]
paddq xmm4, xmm7
pand xmm3, xmm0
mov rdx, rbx
inc rbx
and rcx, rax
imul rdi, rax
dec rcx
and rcx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rcx]
mov rbx, rax
and rdi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rcx]
setl dil
psubq xmm7, xmm7
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rdi]
por xmm3, xmm1
and rdi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
sbb rsi, rdi
pcmpeqd xmm3, xmm1
and rdi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdi]
sub rsi, rdi
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx
imul rsi, rdi
pextrq rbx, xmm1, 0
setb cl
and rbx, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
por xmm1, xmm0
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
not rdx
imul rsi, rcx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx
pxor xmm4, xmm0
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
pxor xmm5, xmm7
setb dil
and rsi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdi
pand xmm2, xmm2
imul rsi, rbx
sbb rcx, rax
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax
and rdi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
