.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rax
neg rdx
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rsi
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rsi]
test rdx, rsi
neg rdi
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
test rcx, rcx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rax
setb bl
and rcx, rcx
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rdx
setnz al
and rax, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
