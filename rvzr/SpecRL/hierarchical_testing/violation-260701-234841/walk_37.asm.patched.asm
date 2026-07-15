.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111110 # instrumentation
movsx rbx, byte ptr [r14 + rsi]
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
and rbx, 0b1111111111110 # instrumentation
movzx rdi, byte ptr [r14 + rbx]
# mem access: [33] 0x1034 cl 0:52 | [83] 0x1034 cl 0:52
xor rdi, rax
and rax, 0b1111111100000 #
sub rdi, rax
and rbx, 0b1111111111100 # instrumentation
mov qword ptr [r14 + rbx], rax
# mem access: [33] 0x1034 cl 0:52 | [83] 0x1034 cl 0:52
adc rdx, rcx
and rbx, 0b1111111111110 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
# mem access: [33] 0x1034 cl 0:52 | [83] 0x1034 cl 0:52
sub rbx, rbx
adc rbx, rdx
and rbx, 0b1111111111110 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx]
# mem access: [33] 0x1608 cl 24:8 | [83] 0x1608 cl 24:8
and rsi, 0b1111111111110 # instrumentation
mul qword ptr [r14 + rsi]
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
and rdx, 0b1111111111100 # instrumentation
movsx rdx, byte ptr [r14 + rdx]
# mem access: [33] 0x1594 cl 22:20 | [83] 0x1594 cl 22:20
and rbx, 0b1111111100000 #
lea rax, qword ptr [rbx + rax + 1]
lfence
and rdi, 0b1111111100000 # instrumentation
xchg qword ptr [r14 + rdi], rax
# mem access: [33] 0x1000-0x1000 cl 0:0 | [83] 0x1000-0x1000 cl 0:0
xor rdx, rdx
and rdx, 0b1111111111100 # instrumentation
movsx rax, byte ptr [r14 + rdx]
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
and rdi, 0b1111111111100 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
# mem access: [33] 0x1000 cl 0:0 | [83] 0x1000 cl 0:0
adc rax, rbx
and rax, 0b1111111110000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
# mem access: [33] 0x1900-0x1900 cl 36:0 | [83] 0x1900-0x1900 cl 36:0
and rax, 0b1111111100000 # instrumentation
lock inc qword ptr [r14 + rax]
# mem access: [33] 0x1000-0x1000 cl 0:0 | [83] 0x1000-0x1000 cl 0:0
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
