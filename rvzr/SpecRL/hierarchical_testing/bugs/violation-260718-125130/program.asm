.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -124 # instrumentation
mov rbx, 4152
imul rdi, rax
mov rbx, 6768
jnp .bb_0.1
jmp .exit_0
.bb_0.1:
mov rcx, 464
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
