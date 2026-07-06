.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
sbb rdi, rcx 
setnz sil 
not rdi 
sbb rdi, rax 
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rcx 
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx] 
setb al 
lea rcx, qword ptr [rsi + rcx + 1] 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rcx 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rbx] 
setl al 
or rdx, rcx 
and rdi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rdi] 
not rax 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
or rax, rcx 
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdx] 
mov rsi, 664 
dec rdi 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rcx] 
imul rdx, rbx 
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi] 
imul rbx, rax 
dec rsi 
dec rsi 
adc rax, rdi 
sub rdx, rbx 
setz dl 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rsi 
setz al 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdi 
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
