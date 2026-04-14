.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -43 # instrumentation
add rax, rsi 
and rsi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rsi 
sbb rax, 5 
sbb rbx, 4 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], 3 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 0 
add rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
cmp rdx, rcx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rax, 4 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
