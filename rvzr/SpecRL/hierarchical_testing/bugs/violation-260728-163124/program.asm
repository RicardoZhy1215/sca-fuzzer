.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rbx, xmm2, 0
and rdx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdx]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
pxor xmm6, xmm1
and rdx, rbx
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
movd xmm7, ebx
cmp rbx, rdi
and rdx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdx]
lea rbx, qword ptr [rdx + rbx + 1]
dec rax
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdi]
sub rsi, rdi
setb dl
neg rcx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rdx
movd xmm1, edi
inc rax
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdi
and rdx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx
neg rax
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi]
mov rdx, 8000
pextrq rdx, xmm4, 0
adc rbx, rcx
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
