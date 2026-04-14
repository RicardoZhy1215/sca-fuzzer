.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 5 # instrumentation
add rax, 1 
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rbx 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rsi 
sbb rax, 4 
sbb rcx, 2 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 5 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], 5 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
