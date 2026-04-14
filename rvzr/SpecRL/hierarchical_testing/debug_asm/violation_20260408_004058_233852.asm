.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -59 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi 
add rax, rbx 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
sbb rax, 2 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
mov rax, 3 
sbb rbx, 1 
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx] 
mov rax, rdi 
and rcx, 0b1111111111111 # instrumentation
sbb rbx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 7 
sbb rcx, 5 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 6 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 0 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
sbb rdx, 5 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 5 
and rdi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 3 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
