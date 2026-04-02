.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -1 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rcx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], 7 
jns .bb_0.1 
jmp .exit_0 
.bb_0.1:
cmp rbx, rdx 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdi 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
