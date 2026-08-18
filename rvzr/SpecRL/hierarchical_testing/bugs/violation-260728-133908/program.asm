.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rax, xmm4, 0
and rcx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
neg rcx
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rdx
and rcx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rcx]
pand xmm0, xmm4
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rcx
movq xmm0, rcx
lea rsi, qword ptr [rsi + rsi + 1]
pextrq rdx, xmm1, 0
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rbx
pxor xmm4, xmm4
adc rdx, rdi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rcx
lea rax, qword ptr [rsi + rax + 1]
and rdi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdi]
dec rsi
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3504
adc rbx, rdi
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
movq xmm1, rdx
and rdi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdi]
neg rsi
or rsi, rsi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi
adc rbx, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
