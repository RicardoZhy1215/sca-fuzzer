.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
cmp rsi, rbx
and rdx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdx]
setz al
and rcx, rsi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
inc rcx
movq xmm0, rbx
xor rsi, rbx
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
setb dil
and rdx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdx]
pextrq rdx, xmm7, 0
psubq xmm1, xmm3
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rsi
and rsi, rsi
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rsi
neg rsi
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
xor rdi, rsi
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
sbb rbx, rdi
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
imul rbx, rcx
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi]
cmp rcx, rbx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
dec rbx
and rcx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rax
imul rcx, rsi
lea rsi, qword ptr [rsi + rsi + 1]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
setz cl
neg rcx
neg rsi
por xmm0, xmm1
adc rdi, rax
setb cl
and rcx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rcx]
setb cl
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
