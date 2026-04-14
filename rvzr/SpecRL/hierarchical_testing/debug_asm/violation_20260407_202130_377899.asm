.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -42 # instrumentation
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 0 
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
xor rcx, rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
sbb rdi, rdx 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdi 
add rsi, 6 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 5 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], 2 
add rax, 2 
jnp .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
add rsi, 4 
and rdi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 5 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 5 
mov rbx, 3 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
