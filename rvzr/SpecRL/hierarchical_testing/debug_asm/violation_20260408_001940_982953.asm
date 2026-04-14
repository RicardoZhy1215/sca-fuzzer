.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 76 # instrumentation
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
jnbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rdi] 
add rax, 0 
add rbx, rcx 
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
