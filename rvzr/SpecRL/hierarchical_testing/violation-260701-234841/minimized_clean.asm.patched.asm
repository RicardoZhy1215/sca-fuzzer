.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx]
and rcx, 0b1111111110000 #
xor rdi, rax
nop dword ptr [rax + 0xff]  # 7 b
and rax, 0b1111111110000 #
and rdi, 0b1111111110000 #
nop qword ptr [rax + 1]  # 5 b
sub rdi, rax
nop dword ptr [rax + 0xff]  # 7 b
nop dword ptr [rax + 0xff]  # 7 b
setb dil
nop dword ptr [rax + 0xff]  # 7 b
and rbx, 0b1111111111110 # instrumentation
mov qword ptr [r14 + rbx], rax
adc rdx, rcx
nop dword ptr [rax + 0xff]  # 7 b
nop dword ptr [rax + 0xff]  # 7 b
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
sub rbx, rbx
nop dword ptr [rax + 0xff]  # 7 b
adc rbx, rdx
and rdx, 0b1111111111110 #
nop qword ptr [rax + 0xff]  # 8 b
and rcx, 0b1111111111110 #
nop qword ptr [rax + 1]  # 5 b
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
nop dword ptr [rax + 0xff]  # 7 b
and rdx, 0b1111111111110 # instrumentation
movsx rdx, byte ptr [r14 + rdx]
and rbx, 0b1111111110000 #
nop qword ptr [rax + 1]  # 5 b
lea rax, qword ptr [rbx + rax + 1]
and rsi, 0b1111111111110 #
nop qword ptr [rax]  # 4 b
nop dword ptr [rax]  # 3 b
and rdi, 0b1111111110000 # instrumentation
xchg qword ptr [r14 + rdi], rax
xor rdx, rdx
and rdx, 0b1111111111110 # instrumentation
movsx rax, byte ptr [r14 + rdx]
and rdi, 0b1111111111110 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
nop dword ptr [rax + 0xff]  # 7 b
nop qword ptr [rax]  # 4 b
and rdi, 0b1111111110000 #
nop qword ptr [rax + 1]  # 5 b
nop qword ptr [rax]  # 4 b
adc rax, rbx
nop dword ptr [rax]  # 3 b
nop dword ptr [rax]  # 3 b
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
and rax, 0b1111111110000 # instrumentation
lock inc qword ptr [r14 + rax]
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
