.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
dec rbx
paddq xmm0, xmm2
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rax
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
adc rbx, rdi
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi
neg rbx
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
setb dl
xor rsi, rdi
dec rdx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi
and rcx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rcx]
neg rdx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
setb bl
and rax, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
adc rbx, rbx
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rsi
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5848
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
mov rbx, rdx
and rcx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rcx]
dec rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi]
setb sil
pextrq rcx, xmm4, 0
movd xmm0, eax
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
mov rbx, 2808
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rcx
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdi
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx
pxor xmm0, xmm0
setb bl
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
neg rsi
adc rbx, rcx
and rdx, rdx
dec rsi
setnz al
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rax]
setb sil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
