.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
paddq xmm5, xmm6
imul rbx, rcx
and rcx, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rcx]
pextrq rbx, xmm2, 0
mov rax, 4264
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4632
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax
mov rcx, rsi
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rsi]
not rcx
neg rsi
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx
xor rsi, rsi
and rax, rax
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rbx
pxor xmm1, xmm1
pxor xmm3, xmm1
paddq xmm6, xmm4
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx]
setl sil
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
neg rsi
sub rsi, rbx
and rbx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rbx]
paddq xmm0, xmm4
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
imul rcx, rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
