.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdi]
mov rsi, rdx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
adc rdx, rcx
and rcx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rcx]
xor rbx, rcx
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rcx
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdx
setb sil
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
mov rsi, rdi
sub rcx, rcx
setb cl
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi
mov rdx, rcx
cmp rsi, rdx
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
setb dl
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rcx
mov rsi, rbx
sub rcx, rcx
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rcx
pextrq rsi, xmm7, 0
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rax]
movq xmm3, rcx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rsi, rcx
xor rbx, rax
and rdi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdi]
pcmpeqd xmm1, xmm3
and rax, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdi
setb sil
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rsi]
not rsi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
mov rbx, rcx
cmp rcx, rcx
and rax, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rax]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
mov rsi, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
