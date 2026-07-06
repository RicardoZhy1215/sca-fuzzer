.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rsi, 4512
pcmpeqd xmm2, xmm0
and rsi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rsi]
sbb rsi, rbx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
and rsi, rdi
adc rbx, rax
and rcx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rcx]
mov rcx, 5184
imul rdx, rsi
dec rax
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
cmp rcx, rsi
setb cl
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
lea rdi, qword ptr [rsi + rdi + 1]
cmp rcx, rdi
and rdx, rbx
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rsi]
mov rcx, 4768
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
jmp .bb_0.1
.bb_0.1:
pxor xmm7, xmm5
test rbx, rsi
and rax, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rax]
setl cl
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7696
and rdi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rdi]
not rdi
setl sil
mov rcx, rdx
and rcx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rcx]
setl dl
or rax, rdi
setz bl
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax
neg rbx
pextrq rsi, xmm6, 0
sbb rbx, rbx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
and rax, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
