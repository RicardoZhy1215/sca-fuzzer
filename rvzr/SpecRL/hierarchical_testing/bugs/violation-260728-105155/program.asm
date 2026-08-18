.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
paddq xmm0, xmm4
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax
pextrq rcx, xmm0, 0
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rcx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rbx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4304
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
neg rcx
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
adc rsi, rax
and rbx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax
paddq xmm5, xmm1
imul rbx, rax
and rax, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rbx]
paddq xmm4, xmm2
and rcx, rsi
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rsi
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rcx
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
imul rcx, rax
setb cl
and rbx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rbx]
setb cl
mov rcx, 5184
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
