.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
lfence
cmovbe rdx, qword ptr [r14 + rdx]
lfence
setnz al
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
or rdi, 1 # instrumentation
lfence
clc  # instrumentation
lfence
cmovnbe rdi, qword ptr [r14 + rsi]
lfence
adc rsi, rcx
lfence
sub rcx, rdx
lfence
dec rsi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
movsx rdx, byte ptr [r14 + rsi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovnle rdx, qword ptr [r14 + rdi]
lfence
not rsi
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rdx], rcx
lfence
mov rcx, rsi
lfence
cmp rdx, rdi
lfence
or rdx, rax
lfence
and rcx, rax
lfence
and rax, 0b1111111111111 # instrumentation
lfence
or rdi, 1 # instrumentation
lfence
cmovnz rdi, qword ptr [r14 + rax]
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmovle rcx, qword ptr [r14 + rdx]
lfence
setl bl
lfence
setl bl
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
or rsi, 1 # instrumentation
lfence
cmovnz rsi, qword ptr [r14 + rdx]
lfence
setl dil
lfence
and rax, 0b1111111111000 # instrumentation
lfence
xchg qword ptr [r14 + rax], rdi
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rcx], rdi
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmp rcx, rcx # instrumentation
lfence
cmovz rcx, qword ptr [r14 + rdx]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
cmovns rsi, qword ptr [r14 + rax]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock xor qword ptr [r14 + rax], rbx
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rax], rbx
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
