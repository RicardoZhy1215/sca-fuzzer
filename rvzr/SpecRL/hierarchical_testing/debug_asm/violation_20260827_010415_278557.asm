.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5400 
inc rbx 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rcx] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
setb dil 
and rsi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2632 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
