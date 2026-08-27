.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rsi
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
setz bl
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
mov rbx, rcx
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rsi
inc rax
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdx
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
