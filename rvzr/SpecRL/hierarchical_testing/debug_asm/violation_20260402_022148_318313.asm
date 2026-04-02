.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 69 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
xor rbx, rbx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 0 
add rdx, rbx 
xor rsi, rax 
mov rcx, rdx 
add rcx, 1 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
xor rax, rdi 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rax 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 4 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rdi, rdx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
mov rbx, rdx 
cmp rax, rbx 
xor rbx, rdi 
add rbx, rdi 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
cmp rbx, rcx 
add rdi, 0 
mov rdi, rcx 
mov rcx, rax 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rbx 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
xor rax, rbx 
xor rax, rbx 
xor rdx, rdx 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
xor rcx, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
