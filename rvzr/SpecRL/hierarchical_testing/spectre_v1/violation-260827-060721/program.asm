.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi
setz al
or rcx, rsi
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rbx]
and rdx, rdx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rbx]
test rdi, rbx
and rsi, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rsi]
setnz cl
setnz cl
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax
inc rbx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
