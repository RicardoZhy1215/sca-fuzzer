.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
adc rax, rsi
pextrq rdi, xmm3, 0
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rcx]
inc rbx
and rdi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rbx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx
and rsi, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rsi]
pxor xmm2, xmm4
pextrq rbx, xmm3, 0
and rbx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rbx]
test rbx, rax
movq xmm7, rax
pcmpeqd xmm0, xmm4
not rdi
sub rdi, rsi
not rax
pcmpeqd xmm2, xmm1
psubq xmm0, xmm5
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
or rcx, rcx
por xmm2, xmm4
neg rax
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx]
setnz dl
xor rdi, rdx
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rcx
movd xmm1, eax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
