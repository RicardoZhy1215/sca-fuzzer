.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 103 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3112
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
sub rdi, rax
mov rax, rsi
or rsi, rbx
mov rdi, rsi
setnz sil
and rcx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdi
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi]
setnz sil
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rax
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1392
and rax, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rax]
mov rax, 7752
inc rcx
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi
and rax, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
and rdx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdx]
and rax, rax
and rbx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx
sbb rax, rdi
mov rbx, rdx
and rax, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rcx]
neg rsi
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rax
dec rdx
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi]
test rbx, rax
setz dil
sbb rbx, rdi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
