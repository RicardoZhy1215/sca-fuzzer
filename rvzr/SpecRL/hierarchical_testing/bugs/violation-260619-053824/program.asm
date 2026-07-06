.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rbx, rsi
and rcx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rcx]
xor rax, rbx
setb al
neg rdx
pxor xmm1, xmm2
pextrq rcx, xmm5, 0
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rdi
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx]
sbb rax, rdi
and rsi, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rsi]
setz bl
not rax
not rbx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rdx
not rbx
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdi
lea rcx, qword ptr [rsi + rcx + 1]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax]
jmp .bb_0.1
.bb_0.1:
setb al
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx]
movq xmm7, rax
and rax, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
mov rdi, rax
or rax, rcx
inc rax
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2312
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdi
mov rbx, 8112
and rdi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi
pextrq rcx, xmm3, 0
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
