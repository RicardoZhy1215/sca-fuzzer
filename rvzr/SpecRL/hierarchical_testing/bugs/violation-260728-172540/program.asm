.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdi
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rcx
and rdx, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rdx]
dec rdx
movd xmm1, ebx
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
dec rsi
inc rax
mov rsi, rbx
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
and rsi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rsi]
dec rbx
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx
pmuludq xmm5, xmm1
pmuludq xmm6, xmm0
pmuludq xmm4, xmm0
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rbx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
setl al
setb bl
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
setl sil
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7240
sub rax, rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3576
mov rdx, 7792
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdx]
neg rax
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
pextrq rax, xmm7, 0
and rdi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdi]
and rbx, rsi
sub rbx, rcx
cmp rdx, rdx
sbb rax, rdx
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
