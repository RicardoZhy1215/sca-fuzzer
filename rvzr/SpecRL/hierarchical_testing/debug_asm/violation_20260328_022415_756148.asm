.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -106 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
xor rcx, rax 
xor rcx, rax 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rsi, rdx 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
xor rdi, rcx 
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx] 
mov rsi, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
