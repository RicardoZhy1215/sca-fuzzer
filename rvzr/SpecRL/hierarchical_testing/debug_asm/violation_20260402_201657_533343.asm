.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 89 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
add rdx, rbx 
add rdx, rax 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rdi 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rax, rcx 
cmp rdx, rbx 
cmp rsi, rdx 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
