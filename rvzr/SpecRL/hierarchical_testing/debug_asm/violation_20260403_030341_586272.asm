.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -60 # instrumentation
xor rbx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
xor rdx, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cmp rax, rbx 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
mov rbx, rcx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rdx, rax 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
xor rsi, rsi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cmp rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
mov rax, rbx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
