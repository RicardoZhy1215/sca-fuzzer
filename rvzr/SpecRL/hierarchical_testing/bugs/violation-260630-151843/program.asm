.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdi]
mov rbx, rdx
and rdx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdi]
psubq xmm2, xmm3
and rsi, rdi
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
lea rbx, qword ptr [rbx + rbx + 1]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rcx
pextrq rsi, xmm2, 0
xor rdx, rdx
setb al
setl bl
and rdi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx
paddq xmm0, xmm1
pcmpeqd xmm2, xmm2
jmp .bb_0.1
.bb_0.1:
inc rdi
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rax]
movq xmm0, rbx
test rdx, rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6656
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rcx
xor rdx, rsi
and rsi, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rsi]
pcmpeqd xmm4, xmm2
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
pextrq rax, xmm7, 0
sub rdi, rdx
not rsi
pand xmm4, xmm3
dec rbx
xor rcx, rbx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx
pextrq rcx, xmm3, 0
and rcx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx
sbb rax, rbx
and rdi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdx]
setb cl
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
