.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rsi]
lea rbx, qword ptr [rdi + rbx + 1]
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rbx
not rbx
sbb rdi, rdi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx
mov rdi, 1656
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax]
adc rdi, rbx
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdx
adc rbx, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
