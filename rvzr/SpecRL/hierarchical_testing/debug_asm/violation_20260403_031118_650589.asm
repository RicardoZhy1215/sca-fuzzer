.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -78 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cmp rbx, rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
add rcx, rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
xor rsi, rdi 
cmp rax, rbx 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
cmp rbx, rdx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cmp rdx, rbx 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
jno .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rbx, rax 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rbx, rcx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
cmp rdx, rax 
cmp rdx, rbx 
xor rax, rsi 
xor rdx, rsi 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rdi, rsi 
mov rbx, rsi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
cmp rdi, rcx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
