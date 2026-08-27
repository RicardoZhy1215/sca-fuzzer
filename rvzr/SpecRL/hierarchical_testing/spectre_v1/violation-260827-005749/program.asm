.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rbx, qword ptr [rcx + rbx + 1]
or rdi, rdx
inc rdi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 760
or rcx, rax
xor rax, rcx
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax]
adc rax, rbx
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
