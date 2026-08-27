.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
test rcx, rcx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdx 
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx] 
imul rdi, rsi 
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rsi 
lea rdi, qword ptr [rax + rdi + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
