.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdx
sub rcx, rdi
and rdx, rsi
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx]
adc rcx, rsi
pxor xmm6, xmm6
xor rdi, rbx
por xmm7, xmm5
pmuludq xmm3, xmm6
setl bl
sbb rcx, rbx
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rcx
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rcx
not rbx
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rax
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
setb sil
pextrq rax, xmm2, 0
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi
paddq xmm6, xmm2
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx]
sub rsi, rcx
pextrq rbx, xmm7, 0
mov rsi, rcx
pxor xmm5, xmm5
setnz sil
and rdx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rcx]
pxor xmm2, xmm1
pextrq rcx, xmm3, 0
and rax, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi]
setl sil
and rcx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rcx]
sub rsi, rdx
por xmm5, xmm7
and rdx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdi
sub rsi, rsi
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rdi
xor rdi, rdx
test rbx, rcx
por xmm7, xmm5
mov rcx, rsi
pxor xmm3, xmm1
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rcx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdi]
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
