.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -117 # instrumentation
add rdx, rax 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rbx, rax 
add rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
mov rbx, rsi 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rsi, rbx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
cmp rcx, rbx 
xor rsi, rbx 
xor rsi, rbx 
mov rsi, rdi 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rdi, rsi 
mov rsi, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
