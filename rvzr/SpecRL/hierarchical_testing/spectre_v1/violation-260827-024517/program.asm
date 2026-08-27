.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setnz al
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
lea rax, qword ptr [rdx + rax + 1]
lea rdx, qword ptr [rbx + rdx + 1]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6048
lea rdi, qword ptr [rdx + rdi + 1]
setnz al
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rdx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax
lea rax, qword ptr [rdx + rax + 1]
xor rax, rsi
and rdx, rcx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
setl dil
setnz sil
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax]
setnz dil
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx
dec rcx
setz cl
adc rdx, rax
setb dil
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
