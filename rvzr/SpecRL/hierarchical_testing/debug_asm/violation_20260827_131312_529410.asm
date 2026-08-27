.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax 
and rax, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rax] 
test rdx, rax 
test rsi, rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdi] 
inc rsi 
lea rcx, qword ptr [rdx + rcx + 1] 
test rbx, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
