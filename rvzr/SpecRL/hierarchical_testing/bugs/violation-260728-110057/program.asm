.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rbx]
por xmm2, xmm6
and rsi, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
sub rdx, rsi
or rsi, rsi
not rcx
neg rdx
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rsi
and rbx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rbx]
not rdx
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
pxor xmm6, xmm4
setb dil
por xmm5, xmm5
not rbx
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
pextrq rcx, xmm2, 0
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx]
movq xmm3, rsi
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rsi
mov rdx, 7864
cmp rax, rdi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
mov rcx, rdi
setb cl
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdi
setb al
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4728
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
paddq xmm7, xmm4
cmp rcx, rax
and rax, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rax]
setnz cl
lea rbx, qword ptr [rdi + rbx + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
