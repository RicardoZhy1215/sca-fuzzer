.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 119 # instrumentation
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi
xor rsi, rax
imul rdi, rax
jns .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
sub rsi, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
