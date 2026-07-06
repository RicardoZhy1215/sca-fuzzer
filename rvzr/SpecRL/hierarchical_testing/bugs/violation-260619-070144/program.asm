.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
adc rdi, rsi
sbb rcx, rdi
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
cmp rsi, rax
movq xmm2, rdi
dec rax
pmuludq xmm1, xmm0
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5616
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
setnz al
and rcx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
psubq xmm5, xmm0
or rcx, rdi
jmp .bb_0.1
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdx]
cmp rcx, rcx
psubq xmm3, xmm7
sub rax, rsi
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
xor rdi, rsi
and rsi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rsi]
pextrq rdi, xmm6, 0
xor rdx, rdi
sbb rbx, rcx
sub rdx, rbx
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
mov rbx, rsi
pextrq rsi, xmm7, 0
setnz al
movq xmm5, rsi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx]
imul rax, rdx
or rax, rbx
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
pand xmm1, xmm0
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
setb al
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
