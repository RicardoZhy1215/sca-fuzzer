.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -85 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rbx 
jnl .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
sbb rcx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx 
add rdx, 0 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
