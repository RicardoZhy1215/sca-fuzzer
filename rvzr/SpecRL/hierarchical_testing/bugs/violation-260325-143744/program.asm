.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 125 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
mov rbx, rcx
xor rax, rdx
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
js .bb_0.1
jmp .exit_0
.bb_0.1:
mov rsi, rax
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
