.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 123 # instrumentation
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdx 
add rdx, rbx 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rdi 
xor rbx, rbx 
xor rbx, rax 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
xor rdi, rcx 
cmp rbx, rax 
xor rax, rcx 
cmp rcx, rdi 
cmp rdx, rcx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
add rbx, rcx 
xor rdx, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rsi, rdi 
xor rdx, rdx 
cmp rdi, rdx 
cmp rax, rsi 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
mov rax, rbx 
add rbx, rax 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
cmp rsi, rax 
add rdx, rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 5 
xor rcx, rcx 
xor rcx, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
