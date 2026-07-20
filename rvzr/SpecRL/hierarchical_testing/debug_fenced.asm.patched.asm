.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 103 # instrumentation
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 3112
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rbx]
lfence
sub rdi, rax
lfence
mov rax, rsi
lfence
or rsi, rbx
lfence
mov rdi, rsi
lfence
setnz sil
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
movsx rdi, byte ptr [r14 + rcx]
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rcx]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock xadd qword ptr [r14 + rdi], rdi
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add rsi, qword ptr [r14 + rbx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmp rsi, rsi # instrumentation
lfence
cmovz rsi, qword ptr [r14 + rsi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
or rbx, 1 # instrumentation
lfence
cmovnz rbx, qword ptr [r14 + rdi]
lfence
setnz sil
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
xchg qword ptr [r14 + rdx], rax
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rax], 1392
lfence
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovns rdi, qword ptr [r14 + rcx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add rax, qword ptr [r14 + rbx]
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock cmpxchg qword ptr [r14 + rsi], rdx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmovnl rdi, qword ptr [r14 + rdx]
lfence
and rax, rax
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmovs rcx, qword ptr [r14 + rbx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], rcx
lfence
sbb rax, rdi
lfence
mov rbx, rdx
lfence
and rax, 0b1111111111111 # instrumentation
lfence
cmp rbx, qword ptr [r14 + rax]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovs rax, qword ptr [r14 + rcx]
lfence
neg rsi
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rax], rax
lfence
dec rdx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmp rsi, qword ptr [r14 + rbx]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
or rdx, 1 # instrumentation
lfence
clc  # instrumentation
lfence
cmovnbe rdx, qword ptr [r14 + rdi]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
