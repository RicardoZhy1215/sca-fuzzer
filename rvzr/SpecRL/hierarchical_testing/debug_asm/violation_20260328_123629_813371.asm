.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 72 # instrumentation
add rsi, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
mov rbx, rdi 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
mov rax, rsi 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 6 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rdx, rdi 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rdx, rbx 
mov rcx, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rbx 
mov rax, rbx 
mov rdx, rax 
xor rax, rsi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
mov rdx, rdi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
mov rbx, rsi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
xor rax, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
