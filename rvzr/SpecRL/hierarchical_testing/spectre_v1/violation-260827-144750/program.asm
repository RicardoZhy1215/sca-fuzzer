.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx]
setnz cl
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx]
mov rdx, 7456
mov rbx, 7712
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax]
lea rbx, qword ptr [rdx + rbx + 1]
setb dil
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
and rsi, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rsi]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, rdi
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdx
sbb rcx, rbx
imul rbx, rdi
setz cl
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx]
setnz cl
and rbx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rbx]
setnz bl
setb dil
and rdx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx
and rdx, rsi
and rdx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rdx]
setnz dl
and rdi, rsi
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
