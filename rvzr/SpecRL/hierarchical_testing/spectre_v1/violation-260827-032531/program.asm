.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rax
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rcx
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
inc rax
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rdi
and rdx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdx]
setnz cl
and rax, rax
and rax, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rax]
mov rax, rsi
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rsi]
setnz sil
setz al
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx]
not rax
mov rcx, 2080
cmp rbx, rsi
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rax
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rax]
setnz cl
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
setl al
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
