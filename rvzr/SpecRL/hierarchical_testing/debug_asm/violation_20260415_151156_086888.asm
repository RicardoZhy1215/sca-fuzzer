.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 121 # instrumentation
cmp rsi, rdi 
and rcx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx 
cmp rdi, rcx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rsi 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 0 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 7 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rcx 
and rdx, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], 5 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
sbb rdi, rbx 
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rcx 
sbb rsi, 4 
sbb rsi, 0 
and rbx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rbx] 
cmp rsi, rcx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 4 
and rbx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 4 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
