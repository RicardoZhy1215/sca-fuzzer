.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
pextrq rdi, xmm2, 0
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
setz dil
or rax, rdi
sub rdx, rsi
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rbx
setz cl
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
lea rax, qword ptr [rax + rax + 1]
jmp .bb_0.1
.bb_0.1:
mov rdi, rbx
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdx
movq xmm3, rdi
mov rcx, 6080
sub rdx, rax
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rax
mov rsi, 2824
pcmpeqd xmm7, xmm7
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rcx
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdx]
lea rax, qword ptr [rcx + rax + 1]
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
