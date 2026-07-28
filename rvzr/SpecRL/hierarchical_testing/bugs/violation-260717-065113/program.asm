.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -28 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rdi
imul rax, rcx
mov rdx, rdi
jns .bb_0.1
jmp .exit_0
.bb_0.1:
mov rbx, 4568
and rdi, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rdi]
test rcx, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
