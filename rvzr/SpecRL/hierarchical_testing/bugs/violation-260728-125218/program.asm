.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax]
mov rbx, 7320
pextrq rbx, xmm1, 0
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx]
setb bl
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 200
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx
imul rcx, rbx
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rax]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
and rbx, rdx
test rcx, rsi
setb sil
movd xmm5, edx
neg rbx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
pxor xmm7, xmm7
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rbx
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx]
neg rsi
pcmpeqd xmm6, xmm5
mov rbx, rcx
setb dl
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
mov rcx, rax
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
setz bl
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx]
neg rax
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
setb cl
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rbx
and rdx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdx]
imul rdx, rsi
pcmpeqd xmm5, xmm0
sbb rax, rax
paddq xmm7, xmm1
setl dil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
