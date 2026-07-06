.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
inc rdi
and rdx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
mov rcx, 5752
imul rsi, rsi
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rdi
setz dl
test rdx, rdx
pextrq rax, xmm6, 0
and rbx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdi
setb sil
and rax, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rax]
test rcx, rax
imul rdx, rbx
pmuludq xmm6, xmm7
adc rcx, rcx
adc rdi, rdx
and rdx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
imul rbx, rax
and rbx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rbx]
por xmm3, xmm6
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx]
dec rdx
and rsi, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rsi]
mov rcx, rax
pxor xmm3, xmm4
setnz dil
lea rcx, qword ptr [rsi + rcx + 1]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi
setl dl
test rdi, rdx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rsi
pand xmm7, xmm7
inc rcx
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rax
and rax, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rax]
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
