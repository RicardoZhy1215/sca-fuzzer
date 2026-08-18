.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
adc rdi, rcx
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx]
cmp rcx, rbx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi
pmuludq xmm6, xmm0
adc rcx, rcx
lea rdx, qword ptr [rdx + rdx + 1]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx
and rcx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rcx]
pxor xmm4, xmm2
pextrq rbx, xmm6, 0
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
paddq xmm3, xmm1
psubq xmm4, xmm2
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx
and rdx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rbx
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx]
por xmm2, xmm6
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
setb cl
not rcx
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdi
sub rcx, rdx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
pextrq rcx, xmm2, 0
lea rcx, qword ptr [rdi + rcx + 1]
pxor xmm4, xmm6
pmuludq xmm4, xmm5
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
por xmm0, xmm1
neg rcx
lea rsi, qword ptr [rdi + rsi + 1]
setl cl
imul rcx, rdi
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
cmp rcx, rax
and rsi, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rsi]
mov rbx, 6
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
