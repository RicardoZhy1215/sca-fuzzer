.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rax
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rsi
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx
test rcx, rsi
xor rbx, rsi
dec rax
imul rdx, rbx
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
mov rdx, 7400
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi]
setnz cl
imul rcx, rsi
and rsi, rax
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi
test rdx, rdi
imul rax, rdi
setz bl
or rcx, rbx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5920
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx]
dec rdx
and rdx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rsi
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx]
mov rdx, 624
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx]
lea rcx, qword ptr [rsi + rcx + 1]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 824
cmp rbx, rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
