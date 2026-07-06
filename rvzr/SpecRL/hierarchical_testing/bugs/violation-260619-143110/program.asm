.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdi
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdi]
not rdi
xor rcx, rdi
and rsi, rdx
setb sil
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
and rdi, rdx
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
psubq xmm2, xmm7
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2360
movq xmm1, rdi
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rdi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdi]
setl sil
or rsi, rbx
por xmm2, xmm3
paddq xmm4, xmm1
mov rdi, 4672
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
not rax
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx]
psubq xmm0, xmm0
jmp .bb_0.1
.bb_0.1:
cmp rdi, rax
paddq xmm4, xmm6
pextrq rcx, xmm7, 0
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
setb dl
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx]
paddq xmm5, xmm1
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
pand xmm1, xmm7
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
setnz al
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
setnz al
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
