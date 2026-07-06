.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rdi, rdi
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rax]
pcmpeqd xmm5, xmm0
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rdi
setnz al
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6112
and rcx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rbx]
pextrq rax, xmm2, 0
and rax, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rax]
jmp .bb_0.1
.bb_0.1:
sbb rsi, rax
and rdi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rax]
pxor xmm1, xmm5
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
lea rsi, qword ptr [rcx + rsi + 1]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
inc rax
setl sil
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdi]
pmuludq xmm4, xmm3
lea rbx, qword ptr [rdi + rbx + 1]
setnz sil
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
xor rsi, rbx
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
setnz al
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdx
setl sil
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
lea rax, qword ptr [rdi + rax + 1]
setz sil
setb al
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi
psubq xmm4, xmm0
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
