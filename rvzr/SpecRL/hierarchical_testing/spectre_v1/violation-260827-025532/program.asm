.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rax
lea rdi, qword ptr [rbx + rdi + 1]
and rsi, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx]
lea rbx, qword ptr [rsi + rbx + 1]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdi
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rcx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx
lea rsi, qword ptr [rdi + rsi + 1]
and rcx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdx
and rax, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rax]
inc rcx
inc rax
cmp rcx, rcx
and rax, rdi
or rax, rbx
and rdx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
neg rcx
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
sub rbx, rbx
adc rbx, rcx
or rdi, rsi
setnz bl
setb al
and rbx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rcx
lea rcx, qword ptr [rdx + rcx + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
