.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rdi, rcx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rsi, qword ptr [r14 + rcx]
lfence
xor rdi, rcx
lfence
xor rsi, rcx
lfence
xor rdi, rcx
lfence
sub rcx, rcx
lfence
xor rdi, rcx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmp rdi, rdi # instrumentation
lfence
cmovz rdi, qword ptr [r14 + rcx]
lfence
sub rdi, rcx
lfence
xor rdi, rcx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rax, qword ptr [r14 + rcx]
lfence
xor rax, rcx
lfence
xor rcx, rcx
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rbx], rcx
lfence
xor rsi, rcx
lfence
sub rbx, rcx
lfence
xor rbx, rcx
lfence
or rbx, 1 # instrumentation
lfence
and rdx, rbx # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rbx
lfence
xor rax, rcx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rax, qword ptr [r14 + rcx]
lfence
xor rdi, rcx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
or rbx, 1 # instrumentation
lfence
cmovnz rbx, qword ptr [r14 + rcx]
lfence
xor rsi, rdx
lfence
xor rdi, rcx
lfence
or rbx, 1 # instrumentation
lfence
and rdx, rbx # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rbx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rbx, qword ptr [r14 + rbx]
lfence
xor rdi, rcx
lfence
sub rsi, rcx
lfence
mov rax, 2736
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rsi], rsi
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdx], rdx
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rax]
lfence
mov rcx, rbx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add rsi, qword ptr [r14 + rdi]
lfence
xor rbx, rcx
lfence
or rdi, 1 # instrumentation
lfence
and rdx, rdi # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rdi
lfence
sbb rsi, rbx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add rsi, qword ptr [r14 + rsi]
lfence
or rax, 1 # instrumentation
lfence
and rdx, rax # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rax
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rcx], rax
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rax, qword ptr [r14 + rbx]
lfence
xor rdi, rcx
lfence
or rdi, 1 # instrumentation
lfence
and rdx, rdi # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rdi
lfence
sbb rdi, rdx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rcx], rdi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add rcx, qword ptr [r14 + rdi]
lfence
mov rcx, 4600
lfence
mov rdi, 360
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rsi]
lfence
xor rcx, rsi
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
