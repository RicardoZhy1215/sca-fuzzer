.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -76 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rdi, rax 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rdx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rsi, rax 
add rax, rsi 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rsi, rdi 
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
cmp rsi, rax 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
cmp rcx, rax 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rsi 
add rax, rdx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cmp rdx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
