.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
dec rbx 
adc rdx, rax 
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdi 
and rcx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rbx, rax 
test rsi, rsi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2856 
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rcx 
lea rcx, qword ptr [rsi + rcx + 1] 
test rdx, rsi 
setb bl 
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
test rsi, rsi 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
setl cl 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
adc rcx, rdi 
setnz dl 
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdi 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rbx 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rdi 
lea rcx, qword ptr [rsi + rcx + 1] 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi 
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rbx] 
and rdi, rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
