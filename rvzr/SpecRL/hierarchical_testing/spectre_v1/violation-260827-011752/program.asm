.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rdi, qword ptr [rdx + rdi + 1]
and rdi, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx
and rcx, rcx
adc rbx, rbx
test rbx, rcx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
lea rdi, qword ptr [rbx + rdi + 1]
setnz al
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rbx
mov rbx, 1704
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
test rcx, rbx
neg rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
