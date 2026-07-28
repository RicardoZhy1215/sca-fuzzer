.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -54 # instrumentation
dec rsi
test rdx, rax
and rax, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rcx
setl dl
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rax
and rax, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
mov rax, 7752
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rsi
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rbx]
lea rcx, qword ptr [rdi + rcx + 1]
and rbx, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
adc rcx, rdi
not rax
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
