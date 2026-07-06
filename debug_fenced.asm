.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, rsi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
movsx rbx, byte ptr [r14 + rsi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
movzx rdi, byte ptr [r14 + rbx]
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rcx]
lfence
xor rdi, rax
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovs rsi, qword ptr [r14 + rsi]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rax], rbx
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock xor qword ptr [r14 + rdi], rdi
lfence
sub rdi, rax
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], rbx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], rsi
lfence
setb dil
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmovs rdi, qword ptr [r14 + rdx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], rax
lfence
adc rdx, rcx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdx], 7888
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovle rbx, qword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
or rax, 1 # instrumentation
lfence
clc  # instrumentation
lfence
cmovnbe rax, qword ptr [r14 + rbx]
lfence
sub rbx, rbx
lfence
cmp rbx, rax
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rbx], rsi
lfence
xor rbx, rbx
lfence
adc rbx, rdx
lfence
and rdx, rdx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdx], 464
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovs rbx, qword ptr [r14 + rcx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
or rsi, 1 # instrumentation
lfence
cmovnz rsi, qword ptr [r14 + rbx]
lfence
adc rdx, rdx
lfence
jmp .bb_0.1
.bb_0.1:
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rsi]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rdi]
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
movsx rdx, byte ptr [r14 + rdx]
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rbx], rsi
lfence
lea rax, qword ptr [rbx + rax + 1]
lfence
setb dl
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add rbx, qword ptr [r14 + rsi]
lfence
setb dl
lfence
xor rcx, rsi
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
xchg qword ptr [r14 + rdi], rax
lfence
xor rdx, rdx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
movsx rax, byte ptr [r14 + rdx]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rbx, qword ptr [r14 + rdi]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovle rdx, qword ptr [r14 + rcx]
lfence
imul rsi, rsi
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rdi], rdx
lfence
imul rdi, rsi
lfence
adc rax, rbx
lfence
adc rdi, rdi
lfence
or rdi, rax
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock cmpxchg qword ptr [r14 + rax], rbx
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rax]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
