.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rsi, rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdi] 
setl al 
setb al 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdx] 
pcmpeqd xmm5, xmm2 
and rdx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi 
and rcx, rax 
and rdx, rbx 
jmp .bb_0.1 
.bb_0.1:
xor rsi, rdx 
pcmpeqd xmm0, xmm6 
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi 
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi] 
and rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
setb al 
and rcx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
