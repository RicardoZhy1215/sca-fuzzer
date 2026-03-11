.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -104 # instrumentation
add rbx, rdx 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 7 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 5 
add rcx, 0 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 5 
cmp rbx, rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7 
add rsi, 1 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4 
add rdi, 6 
add rcx, rdx 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rdi, 5 
mov rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 1 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3 
add rsi, 2 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7 
add rdx, 0 
cmp rsi, rbx 
mov rdx, rbx 
cmp rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 4 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 3 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7 
add rsi, rdi 
add rdx, 5 
add rdi, 7 
add rcx, 3 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 0 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
