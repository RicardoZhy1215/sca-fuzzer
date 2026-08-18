.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi
and rbx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rsi
movq xmm0, rcx
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdx]
setz sil
and rcx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rcx]
movd xmm5, ebx
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
setz bl
psubq xmm3, xmm2
setz al
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 128
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
setl sil
movq xmm0, rcx
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
pextrq rsi, xmm2, 0
mov rdx, 3232
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rsi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
setz bl
pxor xmm7, xmm0
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
pxor xmm3, xmm4
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
