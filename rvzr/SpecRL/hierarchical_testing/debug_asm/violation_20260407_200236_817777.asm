.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -113 # instrumentation
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
add rsi, rdx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rsi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 3 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 1 
mov rcx, 5 
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
