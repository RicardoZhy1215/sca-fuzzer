.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax] 
pmuludq xmm4, xmm5 
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rcx] 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
lea rsi, qword ptr [rcx + rsi + 1] 
and rsi, rdx 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rbx 
setnz dil 
setnz sil 
lea rsi, qword ptr [rax + rsi + 1] 
pand xmm5, xmm1 
and rsi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx 
and rdi, rbx 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx 
xor rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
cmp rsi, rax 
and rax, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rcx 
dec rsi 
psubq xmm1, xmm5 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
and rdx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rcx 
and rdx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdx] 
not rsi 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax 
mov rax, 5288 
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rcx 
setnz sil 
setb sil 
dec rax 
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rbx] 
or rsi, rbx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6328 
por xmm4, xmm5 
setnz sil 
dec rsi 
mov rsi, rdi 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
