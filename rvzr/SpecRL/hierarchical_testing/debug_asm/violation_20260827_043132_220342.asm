.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rax 
and rsi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
