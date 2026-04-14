.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 127 # instrumentation
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 6 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
cmp rsi, rax 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 0 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 0 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 4 
sbb rdx, rbx 
add rbx, 4 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 5 
add rcx, 5 
add rbx, 1 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 3 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 4 
add rdi, 3 
and rdi, 0b1111111111111 # instrumentation
sbb rcx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 7 
jnle .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
sbb rdi, rcx 
mov rcx, 3 
xor rbx, rcx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 3 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
mov rbx, rsi 
sbb rcx, rdx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
xor rsi, rdx 
xor rbx, rdi 
mov rbx, 5 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
