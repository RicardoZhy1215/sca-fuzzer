.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
xor rsi, rcx 
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax] 
psubq xmm0, xmm7 
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx] 
psubq xmm1, xmm6 
setb sil 
mov rsi, rbx 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rbx] 
and rdi, rbx 
dec rsi 
and rdi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx] 
por xmm4, xmm5 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax] 
setb sil 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
sbb rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rax] 
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi 
mov rsi, 5776 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4000 
dec rsi 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi 
and rcx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rcx 
mov rsi, 6064 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rbx] 
pxor xmm7, xmm5 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rax 
not rax 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
movq xmm2, rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
