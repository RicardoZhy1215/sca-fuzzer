.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 992
and rdx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rdx]
mov rsi, rdi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
sbb rax, rcx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi]
lea rsi, qword ptr [rdx + rsi + 1]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5056
sbb rcx, rcx
lea rdx, qword ptr [rdi + rdx + 1]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
sbb rcx, rdi
inc rsi
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
movq xmm0, rdx
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rbx
psubq xmm2, xmm6
and rbx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rbx]
dec rax
sbb rbx, rdx
cmp rsi, rcx
mov rsi, rcx
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdx
adc rsi, rcx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
adc rsi, rdx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
pextrq rax, xmm6, 0
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi
adc rbx, rdi
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
neg rbx
adc rbx, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
