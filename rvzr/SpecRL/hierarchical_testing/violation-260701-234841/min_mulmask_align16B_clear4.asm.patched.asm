.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi]
# mem access: [46] 0x1c77 cl 49:55 | [96] 0x1c77 cl 49:55
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx]
# mem access: [46] 0x105b cl 1:27 | [96] 0x105b cl 1:27
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
# mem access: [46] 0x105a cl 1:26 | [96] 0x105a cl 1:26
adc rdx, rcx
nop dword ptr [rax + 0xff]  # 7 b
nop dword ptr [rax + 0xff]  # 7 b
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx]
# mem access: [46] 0x105a cl 1:26 | [96] 0x105a cl 1:26
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
# mem access: [46] 0x1fe1 cl 63:33 | [96] 0x1fe1 cl 63:33
and rsi, 0b1111111110000 # instrumentation
mul qword ptr [r14 + rsi]
# mem access: [46] 0x219f cl 6:31 | [96] 0x219f cl 6:31
nop dword ptr [rax + 0xff]  # 7 b
and rdx, 0b1111111111110 # instrumentation
movsx rdx, byte ptr [r14 + rdx]
# mem access: [46] 0x17d8 cl 31:24 | [96] 0x17d8 cl 31:24
and rbx, 0b1111111110000 #
nop qword ptr [rax + 1]  # 5 b
lea rax, qword ptr [rbx + rax + 1]
and rsi, 0b1111111111110 #
nop qword ptr [rax]  # 4 b
nop dword ptr [rax]  # 3 b
and rdi, 0b1111111110000 # instrumentation
xchg qword ptr [r14 + rdi], rax
# mem access: [46] 0x1000-0x1000 cl 0:0 | [96] 0x1000-0x1000 cl 0:0
xor rdx, rdx
and rdx, 0b1111111111110 # instrumentation
movsx rax, byte ptr [r14 + rdx]
# mem access: [46] 0x1000 cl 0:0 | [96] 0x1000 cl 0:0
and rdi, 0b1111111111110 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
# mem access: [46] 0x1000 cl 0:0 | [96] 0x1000 cl 0:0
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
# mem access: [46] 0x2640-0x2640 cl 25:0 | [96] 0x2640-0x2640 cl 25:0
and rax, 0b1111111110000 # instrumentation
lock inc qword ptr [r14 + rax]
# mem access: [46] 0x2e90-0x2e90 cl 58:16 | [96] 0x2e90-0x2e90 cl 58:16
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
