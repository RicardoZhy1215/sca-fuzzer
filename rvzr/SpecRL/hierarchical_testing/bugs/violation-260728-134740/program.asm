.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
setb dil
lea rbx, qword ptr [rbx + rbx + 1]
and rbx, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rbx]
pextrq rbx, xmm7, 0
setz sil
por xmm6, xmm5
adc rcx, rax
setb cl
lea rbx, qword ptr [rcx + rbx + 1]
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rcx
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rcx
and rax, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rcx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx
cmp rax, rax
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rbx
por xmm4, xmm7
setb cl
pextrq rsi, xmm7, 0
adc rbx, rax
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
neg rsi
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
dec rdx
adc rdx, rcx
dec rsi
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdi
setb sil
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
adc rsi, rcx
dec rbx
por xmm6, xmm7
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
not rbx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
neg rsi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
