.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
por xmm2, xmm7
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
setz bl
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
pxor xmm6, xmm4
lea rsi, qword ptr [rdx + rsi + 1]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdx]
neg rcx
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
cmp rax, rbx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rbx
paddq xmm6, xmm4
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdi
xor rbx, rbx
neg rbx
pextrq rsi, xmm6, 0
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdi
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rbx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx
movd xmm6, ebx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdx
and rbx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
neg rbx
adc rdx, rcx
and rsi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rbx]
adc rbx, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
