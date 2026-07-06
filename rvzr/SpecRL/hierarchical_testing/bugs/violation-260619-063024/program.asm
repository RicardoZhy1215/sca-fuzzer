.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
neg rax
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdi
cmp rbx, rbx
pand xmm0, xmm7
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rbx
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdi
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
cmp rcx, rbx
pand xmm3, xmm7
pmuludq xmm4, xmm3
pextrq rcx, xmm4, 0
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rbx
and rcx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rcx]
pextrq rdx, xmm5, 0
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
setb bl
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rsi]
neg rsi
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
xor rbx, rcx
cmp rcx, rdx
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rax]
movq xmm4, rsi
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 296
and rbx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rbx
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rsi]
not rdi
and rsi, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rsi]
paddq xmm4, xmm6
por xmm2, xmm2
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
