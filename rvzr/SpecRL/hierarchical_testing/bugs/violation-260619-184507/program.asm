.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rsi]
setnz sil
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rcx
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
setb dil
psubq xmm4, xmm4
mov rsi, 3944
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
pand xmm4, xmm0
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
setnz sil
mov rdi, 1552
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
dec rsi
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx
pxor xmm5, xmm3
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
setnz al
pextrq rdi, xmm5, 0
jmp .bb_0.1
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rsi]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
pxor xmm4, xmm5
and rdx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdx]
and rsi, rax
and rdi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rcx
setb sil
setnz bl
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
pxor xmm4, xmm3
pextrq rcx, xmm1, 0
setl al
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
pxor xmm4, xmm5
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
or rsi, rbx
setnz bl
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 480
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx
and rcx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
