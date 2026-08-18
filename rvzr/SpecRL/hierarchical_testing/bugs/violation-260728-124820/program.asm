.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rsi, qword ptr [rdx + rsi + 1]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5688
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
setb bl
mov rbx, rsi
inc rsi
setb cl
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax
not rsi
neg rsi
setnz bl
setb cl
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
neg rbx
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
pextrq rax, xmm1, 0
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rcx, rdi
pxor xmm0, xmm6
and rcx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rcx
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
pand xmm6, xmm1
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdi]
adc rcx, rbx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
sub rbx, rdx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx
setnz bl
inc rcx
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
paddq xmm4, xmm0
not rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx
and rdi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdi]
setz bl
or rcx, rbx
pmuludq xmm0, xmm0
setb sil
imul rbx, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
