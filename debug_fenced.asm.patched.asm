.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111110 # instrumentation
lfence
movsx rbx, byte ptr [r14 + rsi]
lfence
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
and rbx, 0b1111111111110 # instrumentation
lfence
movzx rdi, byte ptr [r14 + rbx]
lfence
# mem access: [33] 0x1034 cl 0:52 | [83] 0x1034 cl 0:52
xor rdi, rax
lfence
and rax, 0b1111111100000 #
lfence
sub rdi, rax
lfence
and rbx, 0b1111111111100 # instrumentation
lfence
mov qword ptr [r14 + rbx], rax
lfence
# mem access: [33] 0x1034 cl 0:52 | [83] 0x1034 cl 0:52
adc rdx, rcx
lfence
and rbx, 0b1111111111110 # instrumentation
lfence
or rax, 1 # instrumentation
lfence
clc  # instrumentation
lfence
cmovnbe rax, qword ptr [r14 + rbx]
lfence
# mem access: [33] 0x1034 cl 0:52 | [83] 0x1034 cl 0:52
sub rbx, rbx
lfence
adc rbx, rdx
lfence
and rbx, 0b1111111111110 # instrumentation
lfence
or rsi, 1 # instrumentation
lfence
cmovnz rsi, qword ptr [r14 + rbx]
lfence
# mem access: [33] 0x1608 cl 24:8 | [83] 0x1608 cl 24:8
and rsi, 0b1111111111110 # instrumentation
lfence
mul qword ptr [r14 + rsi]
lfence
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
and rdx, 0b1111111111100 # instrumentation
lfence
movsx rdx, byte ptr [r14 + rdx]
lfence
# mem access: [33] 0x1594 cl 22:20 | [83] 0x1594 cl 22:20
and rbx, 0b1111111100000 #
lfence
lea rax, qword ptr [rbx + rax + 1]
lfence
and rdi, 0b1111111100000 # instrumentation
lfence
xchg qword ptr [r14 + rdi], rax
lfence
# mem access: [33] 0x1000-0x1000 cl 0:0 | [83] 0x1000-0x1000 cl 0:0
xor rdx, rdx
lfence
and rdx, 0b1111111111100 # instrumentation
lfence
movsx rax, byte ptr [r14 + rdx]
lfence
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
and rdi, 0b1111111111100 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rbx, qword ptr [r14 + rdi]
lfence
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
adc rax, rbx
lfence
and rax, 0b1111111110000 # instrumentation
lfence
lock cmpxchg qword ptr [r14 + rax], rbx
lfence
# mem access: [33] 0x1900-0x1900 cl 36:0 | [83] 0x1900-0x1900 cl 36:0
and rax, 0b1111111100000 # instrumentation
lfence
lfence
lfence
lock inc qword ptr [r14 + rax]
lfence
# mem access: [33] 0x1000-0x1000 cl 0:0 | [83] 0x1000-0x1000 cl 0:0
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
