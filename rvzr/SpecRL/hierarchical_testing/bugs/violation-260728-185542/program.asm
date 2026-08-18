.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rdx
and rdx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
cmp rax, rbx
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx
and rcx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rcx]
sbb rbx, rcx
adc rax, rdx
setb al
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi
and rdi, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdi]
imul rdx, rax
and rax, rbx
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdx]
mov rcx, rax
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx]
por xmm0, xmm6
cmp rsi, rsi
cmp rsi, rdi
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
not rax
and rdi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdx
not rdx
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2864
and rdx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdx]
dec rax
pxor xmm2, xmm2
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx
mov rbx, rsi
pextrq rdx, xmm7, 0
and rdi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdi]
mov rbx, rdi
and rdx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdx]
and rdx, rbx
paddq xmm6, xmm7
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
