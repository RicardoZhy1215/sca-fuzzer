.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rcx
paddq xmm4, xmm5
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
pxor xmm0, xmm7
setnz bl
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
setb sil
setnz sil
and rcx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4376
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
pmuludq xmm3, xmm5
sbb rax, rdx
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rbx
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 160
and rbx, rcx
and rdi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
setb sil
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
setnz dil
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
pmuludq xmm5, xmm2
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
psubq xmm2, xmm7
setb sil
test rdx, rbx
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
