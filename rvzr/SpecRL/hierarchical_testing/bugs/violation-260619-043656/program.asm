.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rbx]
setnz dil
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6312
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rbx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rcx
and rsi, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rsi]
pxor xmm3, xmm1
sub rbx, rbx
imul rcx, rdi
neg rsi
neg rsi
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi
and rcx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rcx]
dec rax
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdx]
sub rsi, rax
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx]
sub rax, rdi
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rbx
and rdi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi
pextrq rsi, xmm2, 0
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx
adc rax, rsi
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
