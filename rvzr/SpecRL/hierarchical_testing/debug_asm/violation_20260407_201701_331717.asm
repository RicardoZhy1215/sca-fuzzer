.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 16 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 4 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 1 
xor rcx, rsi 
and rax, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rax] 
mov rdx, 0 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 0 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
