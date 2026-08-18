.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rbx, rax
pextrq rax, xmm5, 0
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3856
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 256
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
cmp rbx, rdi
and rcx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
