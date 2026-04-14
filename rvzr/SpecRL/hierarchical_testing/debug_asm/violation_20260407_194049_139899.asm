.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 110 # instrumentation
sbb rax, 0 
xor rax, rdx 
and rax, 0b1111111111111 # instrumentation
sbb rbx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
add rdx, 3 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rsi 
mov rbx, 0 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
