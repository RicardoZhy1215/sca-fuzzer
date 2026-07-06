.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rsi, qword ptr [rbx + rsi + 1]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
cmp rdi, rax
paddq xmm4, xmm7
and rdi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdi]
setnz sil
cmp rsi, rax
pextrq rsi, xmm7, 0
setb sil
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
mov rsi, 1800
and rdx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax
and rsi, rax
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
not rsi
lea rdi, qword ptr [rbx + rdi + 1]
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax]
and rax, rsi
jmp .bb_0.1
.bb_0.1:
or rsi, rax
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx]
pxor xmm5, xmm6
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
movq xmm2, rsi
setb al
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx]
not rdx
lea rsi, qword ptr [rax + rsi + 1]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax]
setb al
xor rsi, rbx
pand xmm0, xmm0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
