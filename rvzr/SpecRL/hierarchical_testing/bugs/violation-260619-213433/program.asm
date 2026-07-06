.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx]
por xmm2, xmm0
setb sil
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2328
and rsi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rsi]
cmp rax, rdx
movq xmm4, rcx
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rbx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
setnz bl
setl al
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx]
psubq xmm0, xmm0
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
dec rax
mov rax, 1200
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
and rdi, rbx
and rbx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rbx]
cmp rdi, rbx
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx
jmp .bb_0.1
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1488
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx]
dec rcx
not rax
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5640
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdx
pextrq rbx, xmm7, 0
setnz al
psubq xmm4, xmm5
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 8120
sbb rsi, rax
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rbx
pxor xmm7, xmm6
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rax
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5512
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
