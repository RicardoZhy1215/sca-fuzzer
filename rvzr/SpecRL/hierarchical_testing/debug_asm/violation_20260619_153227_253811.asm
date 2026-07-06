.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pxor xmm3, xmm6 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
setz dil 
setz sil 
and rcx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rcx] 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
pxor xmm3, xmm4 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
paddq xmm2, xmm3 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rcx 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx 
mov rsi, rdi 
or rax, rbx 
psubq xmm3, xmm1 
setnz sil 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax 
setnz sil 
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax] 
setb sil 
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx] 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx] 
and rsi, rbx 
pxor xmm0, xmm5 
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi] 
dec rsi 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
pxor xmm3, xmm5 
pxor xmm1, xmm6 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
mov rsi, rdx 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi 
and rcx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rcx] 
and rcx, rdx 
pand xmm2, xmm0 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi 
and rdi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdi] 
dec rdi 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rcx 
setb sil 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx 
setb sil 
setb sil 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
