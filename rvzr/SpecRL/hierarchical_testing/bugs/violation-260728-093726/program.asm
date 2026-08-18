.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdi
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rax]
psubq xmm3, xmm7
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
pmuludq xmm6, xmm4
pextrq rsi, xmm4, 0
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdx
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdx
paddq xmm6, xmm7
and rax, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
psubq xmm7, xmm4
setl dil
xor rsi, rdi
and rdx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rsi
sbb rcx, rdi
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdx
por xmm1, xmm7
neg rcx
and rsi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rsi]
movq xmm4, rsi
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rsi
imul rcx, rbx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
imul rdi, rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
