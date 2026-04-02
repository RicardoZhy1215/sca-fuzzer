.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 26 # instrumentation
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rdi 
xor rsi, rdi 
add rbx, rcx 
mov rdx, rax 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax 
jb .bb_0.1 
jmp .exit_0 
.bb_0.1:
add rdx, rdi 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rsi 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
