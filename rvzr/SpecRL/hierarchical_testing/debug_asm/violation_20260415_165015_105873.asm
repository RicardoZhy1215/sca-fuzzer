.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -80 # instrumentation
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax] 
add rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 0 
and rdx, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 5 
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi] 
sbb rdi, 3 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 7 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
add rdx, 6 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdx 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5 
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdi] 
mov rsi, 4 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], 4 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
