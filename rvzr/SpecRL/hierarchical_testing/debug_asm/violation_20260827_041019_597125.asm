.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 776 
sbb rcx, rdx 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
lea rax, qword ptr [rdx + rax + 1] 
or rax, rdx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdx] 
setl dl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
