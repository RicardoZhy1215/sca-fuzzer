.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdx
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
mov rsi, 6728
mov rdi, rax
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rbx
and rsi, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rsi], xmm4
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx
pxor xmm7, xmm7
sbb rdi, rdi
and rcx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
and rdi, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdi], xmm1
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx]
xor rsi, rbx
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1536
and rdx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rsi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx]
pmuludq xmm1, xmm5
movq xmm2, rdi
and rdi, 0b1111111110000 # instrumentation
movups xmm3, xmmword ptr [r14 + rdi]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
xor rax, rcx
and rdx, 0b1111111110000 # instrumentation
movups xmmword ptr [r14 + rdx], xmm4
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 520
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax]
and rax, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rax], xmm7
and rsi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx]
sbb rsi, rcx
xor rdi, rdi
sbb rcx, rax
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rcx]
and rdx, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdx], xmm4
movq xmm1, rax
and rdi, 0b1111111110000 # instrumentation
movdqu xmmword ptr [r14 + rdi], xmm2
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
