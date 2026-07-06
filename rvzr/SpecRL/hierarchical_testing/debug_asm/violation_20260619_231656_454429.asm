.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 336 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 224 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3432 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 952 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
movq xmm4, rdx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7352 
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax] 
and rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
or rsi, rdx 
setl dil 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
neg rdx 
sbb rdi, rbx 
jmp .bb_0.1 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7888 
mov rbx, 2672 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4032 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5128 
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
mov rsi, 6768 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4000 
mov rdi, 7640 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5984 
mov rax, 2040 
setb sil 
and rax, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi] 
pxor xmm4, xmm5 
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1728 
and rax, rbx 
movq xmm2, rbx 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
or rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
