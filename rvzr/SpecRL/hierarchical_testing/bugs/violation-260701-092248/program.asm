.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx]
and rdi, rcx
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
setb sil
and rdi, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rdi]
sub rbx, rdi
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
sub rsi, rbx
and rbx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
mov rbx, rax
and rdi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdi]
mov rdx, 4880
sbb rsi, rax
xor rdx, rax
setnz cl
setz cl
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax
imul rdx, rbx
and rsi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rsi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
setz al
and rax, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1152
mov rsi, 3328
and rax, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rcx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rcx
setnz dl
jmp .bb_0.1
.bb_0.1:
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rax
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax
and rdi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx
setb sil
setnz sil
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
sub rax, rdx
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rdi
test rcx, rax
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
or rdx, rax
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
and rbx, 0b1111111111111 # instrumentation
test rcx, rdi
and rax, 0b1111111111111 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
