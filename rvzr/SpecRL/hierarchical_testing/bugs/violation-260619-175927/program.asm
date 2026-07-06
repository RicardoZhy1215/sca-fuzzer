.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx]
psubq xmm2, xmm3
and rcx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
dec rbx
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
pxor xmm6, xmm3
adc rbx, rbx
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
por xmm2, xmm5
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
setnz cl
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
psubq xmm4, xmm0
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx]
jmp .bb_0.1
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
and rsi, rbx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rcx
pextrq rsi, xmm4, 0
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rcx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rcx
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
setnz bl
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdx
psubq xmm1, xmm0
psubq xmm7, xmm0
pextrq rbx, xmm2, 0
test rcx, rdx
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rdx
setnz sil
or rdi, rbx
setb sil
or rsi, rbx
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx
mov rax, 3608
cmp rdi, rbx
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
