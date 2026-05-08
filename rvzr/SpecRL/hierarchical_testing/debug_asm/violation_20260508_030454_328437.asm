.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4168 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi 
mov rcx, 4752 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
lea rdx, qword ptr [rdx + rdx + 1] 
mov rcx, 1688 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
