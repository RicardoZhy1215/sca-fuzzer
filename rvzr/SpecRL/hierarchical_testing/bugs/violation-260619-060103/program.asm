.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
imul rax, rcx
lea rdx, qword ptr [rcx + rdx + 1]
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rax, rbx
and rcx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx]
pand xmm1, xmm3
imul rsi, rcx
not rbx
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi]
test rax, rsi
adc rcx, rdi
setnz dil
cmp rcx, rsi
and rdx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdx]
dec rbx
neg rax
sub rcx, rdx
cmp rax, rbx
jmp .bb_0.1
.bb_0.1:
sub rbx, rcx
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdi
or rdi, rsi
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
setz cl
and rsi, rax
setb dl
and rsi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rax
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx
xor rcx, rsi
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
psubq xmm1, xmm3
pxor xmm3, xmm1
pextrq rax, xmm4, 0
pcmpeqd xmm0, xmm1
psubq xmm1, xmm4
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx
sbb rdx, rsi
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
