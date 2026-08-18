.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rsi, rdi
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rax
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rcx]
pcmpeqd xmm2, xmm7
imul rsi, rdi
por xmm2, xmm6
not rdi
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi]
movd xmm2, edi
and rcx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rax
pmuludq xmm4, xmm7
imul rdx, rax
dec rcx
neg rcx
and rdi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdi]
setz al
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rdi
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdx]
mov rcx, 1160
movd xmm6, esi
pextrq rbx, xmm4, 0
and rax, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rax]
por xmm0, xmm2
not rcx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
dec rsi
sub rcx, rax
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
movd xmm2, edi
sbb rsi, rsi
test rcx, rbx
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rsi
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
not rsi
imul rsi, rax
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
adc rax, rbx
and rdi, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rdi]
psubq xmm7, xmm1
or rsi, rdx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx
and rax, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
setnz bl
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
