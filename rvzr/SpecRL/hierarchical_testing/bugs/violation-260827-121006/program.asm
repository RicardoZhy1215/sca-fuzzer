.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rax
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx
lea rdx, qword ptr [rsi + rdx + 1]
imul rsi, rsi
adc rax, rsi
setl cl
and rdx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rdi]
lea rdi, qword ptr [rax + rdi + 1]
and rax, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx]
lea rbx, qword ptr [rsi + rbx + 1]
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx]
setl dl
test rdx, rax
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx]
lea rdx, qword ptr [rdx + rdx + 1]
imul rax, rsi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rax
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi
sub rsi, rax
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi]
mov rax, rsi
lea rbx, qword ptr [rbx + rbx + 1]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi
test rdx, rdx
adc rcx, rdx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rax
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rbx
imul rax, rax
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdx]
or rbx, rbx
mov rcx, 7752
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi]
lea rcx, qword ptr [rbx + rcx + 1]
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx
setl sil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
