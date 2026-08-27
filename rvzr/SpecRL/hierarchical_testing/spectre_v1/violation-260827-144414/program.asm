.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rax, qword ptr [rax + rax + 1]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdx
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
inc rcx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi
and rbx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1016
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rsi
setnz cl
and rsi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rcx
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
test rbx, rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rax
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5016
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rsi
not rsi
and rbx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
mov rcx, 3656
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rax]
lea rcx, qword ptr [rbx + rcx + 1]
and rax, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rax]
test rcx, rbx
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
