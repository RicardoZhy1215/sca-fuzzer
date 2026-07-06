.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pxor xmm4, xmm0
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
pextrq rbx, xmm2, 0
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rsi
mov rdi, rbx
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rdx
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rcx]
inc rsi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
setb sil
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx]
and rdi, rdx
dec rbx
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rbx
test rbx, rax
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx]
lea rsi, qword ptr [rsi + rsi + 1]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rsi
pextrq rdx, xmm1, 0
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rcx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
sub rcx, rax
mov rsi, rdx
neg rbx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rsi
imul rsi, rdx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
