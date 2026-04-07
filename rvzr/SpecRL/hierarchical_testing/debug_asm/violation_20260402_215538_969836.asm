.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 19 # instrumentation
add rdx, rcx 
add rbx, rdi 
cmp rcx, rdi 
mov rdi, rsi 
cmp rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
cmp rbx, rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rdi, rbx 
cmp rax, rdi 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax 
xor rax, rbx 
add rcx, rax 
add rax, rcx 
add rsi, rdi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
jo .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rdx, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
cmp rsi, rbx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
add rdx, rdi 
xor rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
xor rdi, rax 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
cmp rdx, rdi 
xor rsi, rsi 
add rcx, rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
