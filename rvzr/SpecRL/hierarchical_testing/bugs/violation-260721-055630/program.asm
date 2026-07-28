.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 79 # instrumentation
imul rcx, rcx
js .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
sub rax, rdx
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
