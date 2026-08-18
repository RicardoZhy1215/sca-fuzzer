.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
test rdi, rax
dec rcx
and rdi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
cmp rsi, rdi
and rdx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rdx
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rsi
pand xmm2, xmm2
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
paddq xmm0, xmm0
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx]
movq xmm0, rcx
and rsi, rsi
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rdi
paddq xmm3, xmm0
and rdi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdi]
adc rbx, rax
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
setnz dil
lea rdx, qword ptr [rdi + rdx + 1]
and rcx, rdi
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
pextrq rcx, xmm7, 0
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi]
setz cl
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
imul rcx, rsi
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
and rbx, rdi
and rdi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdi]
neg rsi
and rsi, rax
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
lea rcx, qword ptr [rdi + rcx + 1]
adc rdi, rcx
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
pcmpeqd xmm6, xmm0
and rdi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
