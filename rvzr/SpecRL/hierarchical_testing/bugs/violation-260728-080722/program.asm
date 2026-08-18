.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pxor xmm4, xmm1
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdx
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
imul rax, rdi
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx]
test rax, rbx
dec rax
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
and rax, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rax]
mov rdi, rdx
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi
setz dl
pmuludq xmm2, xmm0
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rcx
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
xor rsi, rsi
pmuludq xmm7, xmm0
pextrq rcx, xmm3, 0
imul rbx, rdi
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
or rax, rdx
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rsi
or rax, rbx
or rsi, rbx
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
sub rax, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
