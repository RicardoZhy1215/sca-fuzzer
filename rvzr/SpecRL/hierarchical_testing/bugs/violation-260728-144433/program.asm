.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rax
and rdi, rdx
dec rcx
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
setb dl
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx
paddq xmm2, xmm6
cmp rax, rdx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
sub rsi, rdx
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rcx]
lea rbx, qword ptr [rdi + rbx + 1]
pextrq rsi, xmm4, 0
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rsi
pxor xmm0, xmm0
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rsi
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rdi
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
lea rdx, qword ptr [rcx + rdx + 1]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rcx
sub rdx, rbx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx
paddq xmm1, xmm5
inc rsi
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdx]
setb sil
cmp rsi, rbx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
neg rsi
inc rbx
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
imul rcx, rax
pextrq rax, xmm3, 0
sub rsi, rsi
pcmpeqd xmm1, xmm7
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rcx]
neg rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
