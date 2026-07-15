.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 3 # instrumentation
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
or rbx, 1 # instrumentation
lfence
cmovnz rbx, qword ptr [r14 + rdi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovns rcx, qword ptr [r14 + rdi]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rcx], rdi
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rdi], rsi
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rbx]
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rdx]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rdi], rsi
lfence
setz cl
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmp rdx, rdx # instrumentation
lfence
cmovz rdx, qword ptr [r14 + rsi]
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rdx], rdi
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rsi], rcx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov rbx, qword ptr [r14 + rdi]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovnle rbx, qword ptr [r14 + rcx]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
add rsi, qword ptr [r14 + rax]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
movzx rsi, byte ptr [r14 + rbx]
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock cmpxchg qword ptr [r14 + rdx], rbx
lfence
and rcx, rbx
lfence
neg rcx
lfence
sub rdx, rbx
lfence
or rdi, 1 # instrumentation
lfence
and rdx, rdi # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rdi
lfence
dec rbx
lfence
lea rdi, qword ptr [rdx + rdi + 1]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rdi]
lfence
setl dil
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmovle rbx, qword ptr [r14 + rbx]
lfence
jle .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
sub rsi, rdx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovle rbx, qword ptr [r14 + rsi]
lfence
xor rsi, rax
lfence
setnz dl
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovns rax, qword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmp qword ptr [r14 + rbx], rbx
lfence
setb bl
lfence
neg rdi
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmovle rsi, qword ptr [r14 + rbx]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
movzx rsi, byte ptr [r14 + rax]
lfence
and rcx, rdi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rdi, qword ptr [r14 + rsi]
lfence
or rsi, rsi
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rcx], rsi
lfence
xor rcx, rbx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmp rax, rax # instrumentation
lfence
cmovz rax, qword ptr [r14 + rsi]
lfence
lea rdx, qword ptr [rcx + rdx + 1]
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rcx]
lfence
and rdx, rbx
lfence
dec rbx
lfence
and rsi, rdi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rdi]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
