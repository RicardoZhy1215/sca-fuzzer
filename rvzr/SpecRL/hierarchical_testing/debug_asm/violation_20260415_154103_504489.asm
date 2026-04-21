.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -67 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
xor rax, rdx 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
jnl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdx 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 1 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
