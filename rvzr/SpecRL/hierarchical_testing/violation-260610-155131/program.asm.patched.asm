.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rdi, rcx
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rcx]
xor rdi, rcx
xor rsi, rcx
xor rdi, rcx
sub rcx, rcx
xor rdi, rcx
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
sub rdi, rcx
xor rdi, rcx
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rcx]
xor rax, rcx
xor rcx, rcx
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
xor rsi, rcx
sub rbx, rcx
xor rbx, rcx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
xor rax, rcx
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rcx]
xor rdi, rcx
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rcx]
xor rsi, rdx
xor rdi, rcx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
xor rdi, rcx
sub rsi, rcx
mov rax, 2736
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
mov rcx, rbx
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
xor rbx, rcx
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
sbb rsi, rbx
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rax
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx]
xor rdi, rcx
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
sbb rdi, rdx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
mov rcx, 4600
mov rdi, 360
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
xor rcx, rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
