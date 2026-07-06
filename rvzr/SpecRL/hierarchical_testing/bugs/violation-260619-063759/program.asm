.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rax, rcx
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rax
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rcx
setb al
setb al
sub rax, rcx
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
neg rax
sbb rdx, rdx
movq xmm4, rsi
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
pextrq rbx, xmm7, 0
and rdx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdx]
imul rax, rdx
pextrq rcx, xmm7, 0
jmp .bb_0.1
.bb_0.1:
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx]
movq xmm6, rsi
or rcx, rsi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rbx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdx
sbb rax, rdx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
setl al
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax
pcmpeqd xmm2, xmm6
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rsi]
setnz al
mov rdi, 5640
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi]
sbb rsi, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
