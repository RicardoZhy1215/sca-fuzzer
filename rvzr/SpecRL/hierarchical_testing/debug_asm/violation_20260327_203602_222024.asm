.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -33 # instrumentation
xor rcx, rax 
mov rax, rdi 
add rdi, rax 
mov rsi, rdi 
xor rdx, rcx 
xor rsi, rbx 
xor rbx, rbx 
xor rsi, rbx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rcx 
jbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdi 
add rax, rsi 
add rax, rcx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
add rcx, rbx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rsi 
cmp rdx, rax 
add rcx, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rcx, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
