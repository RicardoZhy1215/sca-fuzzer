.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setnz bl
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdx]
not rbx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rdx
and rdi, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdx]
pxor xmm3, xmm2
and rdx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdx]
cmp rax, rbx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rdi
and rbx, rdi
pmuludq xmm4, xmm3
sub rax, rcx
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx
pextrq rax, xmm1, 0
dec rsi
sbb rax, rdi
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdi
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rsi]
sub rax, rbx
not rcx
adc rbx, rdi
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdi]
sbb rax, rdx
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
setb al
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rcx
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rdx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdx
cmp rbx, rdi
not rbx
movq xmm1, rcx
and rax, rdx
and rdi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rax
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdx
pextrq rsi, xmm1, 0
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rdi
inc rdi
and rdi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
