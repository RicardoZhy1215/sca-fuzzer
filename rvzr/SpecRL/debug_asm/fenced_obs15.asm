.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
add cl, dl 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rbx 
and rbx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rbx], ecx 
and rax, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rax], ecx 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rsi], bl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
