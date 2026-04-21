.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -10 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 0 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdx 
sbb rax, 0 
cmp rdi, rax 
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
add rcx, rsi 
cmp rsi, rbx 
and rsi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
sbb rsi, 0 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rax 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx 
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
sbb rbx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx 
mov rdx, 6 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 0 
sbb rdx, 2 
xor rsi, rax 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
sbb rsi, 4 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 0 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx 
sbb rbx, 5 
xor rbx, rbx 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 0 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rax 
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
