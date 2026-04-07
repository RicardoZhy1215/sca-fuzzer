.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -123 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdi 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
xor rdx, rcx 
mov rax, rbx 
mov rsi, rdx 
add rax, rcx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rdi, rbx 
cmp rbx, rdx 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
cmp rdx, rax 
xor rdx, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
mov rdx, rcx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
