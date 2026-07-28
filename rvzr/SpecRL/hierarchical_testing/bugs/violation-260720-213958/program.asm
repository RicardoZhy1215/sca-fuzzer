.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 92 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
sub rax, rdx
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
sub rdi, rsi
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rdx]
imul rbx, rax
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
not rdx
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
