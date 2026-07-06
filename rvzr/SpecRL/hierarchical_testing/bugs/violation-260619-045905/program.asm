.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3952
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 384
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rbx]
setl sil
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rax
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rax
pxor xmm1, xmm0
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
setz sil
adc rsi, rdx
xor rdi, rax
and rdi, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
pextrq rdi, xmm4, 0
adc rax, rsi
lea rbx, qword ptr [rsi + rbx + 1]
and rcx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rcx]
sbb rbx, rdi
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx
neg rsi
and rsi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rsi]
dec rdi
neg rsi
mov rbx, 368
and rsi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdi
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi
pextrq rsi, xmm0, 0
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rcx
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi
pmuludq xmm2, xmm1
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx
setb dil
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx
pxor xmm3, xmm4
sbb rsi, rsi
xor rdx, rcx
mov rdi, 5704
mov rsi, rdx
and rcx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rcx]
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
