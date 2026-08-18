.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setb cl
pextrq rcx, xmm7, 0
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx
and rdi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdx
adc rbx, rcx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdi, rcx
setb dil
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdx
neg rsi
and rax, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
inc rsi
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rbx
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
psubq xmm3, xmm2
and rdi, rax
and rdx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdx]
pxor xmm4, xmm2
sbb rbx, rcx
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rcx
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
pxor xmm7, xmm6
not rbx
dec rbx
movd xmm6, edx
setb cl
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rdx
and rdx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdx
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
