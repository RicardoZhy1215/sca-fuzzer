.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rsi, rdi
mov rcx, rax
sbb rbx, rsi
pextrq rax, xmm4, 0
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
adc rsi, rbx
setl al
sbb rcx, rcx
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
sbb rdx, rdi
pxor xmm6, xmm5
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi]
adc rbx, rdi
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rdx]
dec rbx
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
xor rsi, rbx
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
