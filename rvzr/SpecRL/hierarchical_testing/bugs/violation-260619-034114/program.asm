.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rax, rdi
movq xmm3, rdx
xor rdx, rax
and rbx, rdi
test rbx, rcx
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdx
setb sil
and rdi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdi
and rdx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdx]
pxor xmm2, xmm2
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
pextrq rbx, xmm5, 0
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rsi
and rbx, rax
paddq xmm2, xmm7
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdi]
setnz dl
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
mov rbx, 4256
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax]
test rdx, rsi
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
lea rbx, qword ptr [rdi + rbx + 1]
mov rax, rbx
and rbx, rcx
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax
paddq xmm0, xmm5
pxor xmm5, xmm6
and rsi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rsi
setnz sil
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
test rbx, rdx
por xmm3, xmm4
movq xmm0, rbx
inc rsi
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
