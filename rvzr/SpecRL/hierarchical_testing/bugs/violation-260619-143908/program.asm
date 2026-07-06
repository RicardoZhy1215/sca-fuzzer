.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rax, qword ptr [rdx + rax + 1]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rcx
and rdi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdi]
psubq xmm0, xmm3
mov rsi, rcx
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
lea rsi, qword ptr [rax + rsi + 1]
adc rsi, rcx
and rsi, rbx
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx]
and rsi, rax
pextrq rsi, xmm6, 0
psubq xmm2, xmm7
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
setb sil
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rbx
setb sil
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rbx
and rsi, rbx
cmp rsi, rax
jmp .bb_0.1
.bb_0.1:
and rsi, rax
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax]
psubq xmm5, xmm7
setb sil
setz sil
pand xmm2, xmm4
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx
and rsi, rcx
setb sil
xor rsi, rbx
psubq xmm3, xmm4
not rdx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rcx
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
lea rcx, qword ptr [rcx + rcx + 1]
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx]
not rsi
and rdi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi]
movq xmm4, rbx
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
