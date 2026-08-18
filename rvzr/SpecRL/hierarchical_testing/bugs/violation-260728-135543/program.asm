.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rbx, rax
pextrq rax, xmm5, 0
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3856
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 256
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
cmp rbx, rdi
and rcx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
paddq xmm0, xmm7
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
imul rcx, rsi
cmp rdx, rcx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rax
neg rbx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
inc rsi
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
setb sil
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
adc rbx, rcx
pcmpeqd xmm0, xmm5
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rsi
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rcx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
pxor xmm6, xmm0
adc rsi, rsi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi
adc rsi, rsi
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rcx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
setb bl
dec rbx
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
