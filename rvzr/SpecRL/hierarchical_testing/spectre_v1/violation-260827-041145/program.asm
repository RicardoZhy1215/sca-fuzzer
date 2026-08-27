.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rax, rax
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5072
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rax]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rcx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
lea rax, qword ptr [rcx + rax + 1]
and rdx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdx]
lea rax, qword ptr [rbx + rax + 1]
lea rax, qword ptr [rsi + rax + 1]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3816
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
