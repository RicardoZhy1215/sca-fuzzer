.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx
not rax
mov rsi, 3304
not rax
and rdx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdx
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rcx]
setb sil
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
or rsi, rcx
mov rsi, 7224
and rcx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
sbb rdi, rbx
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rcx]
pand xmm0, xmm7
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rax
pmuludq xmm6, xmm5
and rsi, rax
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
or rsi, rax
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
setb sil
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx]
not rsi
not rbx
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
pextrq rsi, xmm6, 0
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdi
neg rax
movq xmm2, rbx
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
