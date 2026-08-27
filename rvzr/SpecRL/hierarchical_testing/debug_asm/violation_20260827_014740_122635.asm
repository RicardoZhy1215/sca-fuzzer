.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax 
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rax] 
lea rax, qword ptr [rdx + rax + 1] 
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
