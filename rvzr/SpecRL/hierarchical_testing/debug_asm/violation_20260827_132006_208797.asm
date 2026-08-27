.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
adc rcx, rbx 
lea rdx, qword ptr [rdx + rdx + 1] 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx 
lea rcx, qword ptr [rbx + rcx + 1] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rdx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 680 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
