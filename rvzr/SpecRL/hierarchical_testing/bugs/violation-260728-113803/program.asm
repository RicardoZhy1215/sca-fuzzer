.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
movq xmm4, rcx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rax
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
psubq xmm0, xmm4
inc rdi
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdi]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
pxor xmm3, xmm0
psubq xmm2, xmm4
pextrq rbx, xmm4, 0
setb cl
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi]
mov rcx, rdx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
por xmm0, xmm7
inc rbx
pmuludq xmm7, xmm0
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax
setb cl
setnz dl
pextrq rbx, xmm1, 0
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rax]
setb cl
setz bl
and rbx, rdx
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rax]
pmuludq xmm0, xmm7
and rax, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdi]
and rcx, rsi
and rax, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdx
neg rbx
and rsi, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rbx]
psubq xmm2, xmm1
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
