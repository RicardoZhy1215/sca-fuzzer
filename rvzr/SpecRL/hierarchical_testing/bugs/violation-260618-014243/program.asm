.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rdx, 1944
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
paddq xmm7, xmm0
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
pcmpeqd xmm4, xmm6
and rcx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7296
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi
or rdx, rcx
and rax, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rax]
test rcx, rax
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rdx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx
mov rdx, 6920
imul rsi, rsi
xor rbx, rsi
lea rcx, qword ptr [rdi + rcx + 1]
and rsi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rsi]
not rdx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx
pand xmm4, xmm3
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
lea rdi, qword ptr [rsi + rdi + 1]
adc rdi, rdi
setl sil
sub rcx, rbx
cmp rax, rdx
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
test rcx, rsi
and rcx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
inc rdi
and rdi, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdi]
inc rcx
mov rsi, rcx
and rbx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rbx]
adc rdx, rbx
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rcx
pextrq rsi, xmm3, 0
pand xmm0, xmm3
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax
mov rdi, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
