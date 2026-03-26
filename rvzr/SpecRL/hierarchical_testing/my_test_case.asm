.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -40 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
add rcx, rdi 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rcx 
xor rsi, rdi 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
cmp rcx, rdx 
mov rsi, rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
mov rdi, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
cmp rax, rdi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rcx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
add rsi, rdi 
mov rcx, rdi 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
