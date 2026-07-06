.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rsi, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rsi]
cmp rdx, rbx
not rdi
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
cmp rdi, rbx
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax]
pmuludq xmm1, xmm1
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx
pand xmm6, xmm0
pextrq rdx, xmm7, 0
and rsi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdi
por xmm5, xmm2
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
setb dl
and rsi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rsi]
cmp rdi, rbx
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rax
setz dil
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
jmp .bb_0.1
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdi]
pmuludq xmm3, xmm1
cmp rdi, rdx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rax]
psubq xmm3, xmm2
and rdx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdx]
por xmm4, xmm2
adc rbx, rsi
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx
sbb rsi, rcx
imul rdi, rbx
pxor xmm1, xmm0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
