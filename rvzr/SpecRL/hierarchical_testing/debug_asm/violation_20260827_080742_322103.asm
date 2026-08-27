.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
sub rbx, rax 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rdi, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rdi] 
test rdi, rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
