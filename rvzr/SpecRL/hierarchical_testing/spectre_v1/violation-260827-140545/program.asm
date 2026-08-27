.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
imul rcx, rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdi
neg rsi
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rsi]
setnz sil
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdi
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
test rcx, rcx
and rax, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rax]
lea rdi, qword ptr [rax + rdi + 1]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rsi
mov rbx, 1968
and rbx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
mov rdx, 4008
and rdx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
