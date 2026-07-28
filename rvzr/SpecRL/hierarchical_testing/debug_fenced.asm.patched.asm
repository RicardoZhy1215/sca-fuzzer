.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
pmuludq xmm0, xmm7
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock xor qword ptr [r14 + rdi], rcx
lfence
mov rsi, 4456
lfence
and rdi, rax
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rcx], rdi
lfence
and rax, 0b1111111111111 # instrumentation
lfence
cmovns rcx, qword ptr [r14 + rax]
lfence
not rsi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rdi], rcx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
movsx rsi, byte ptr [r14 + rcx]
lfence
psubq xmm4, xmm4
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rax], rdi
lfence
lea rbx, qword ptr [rcx + rbx + 1]
lfence
adc rcx, rsi
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rax], 5912
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmovl rdi, qword ptr [r14 + rdx]
lfence
adc rdi, rdx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
movzx rdx, byte ptr [r14 + rbx]
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rsi], rdi
lfence
setb sil
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock cmpxchg qword ptr [r14 + rsi], rdi
lfence
setb bl
lfence
not rdi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
movsx rcx, byte ptr [r14 + rdi]
lfence
mov rax, 4848
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rax]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
movsx rbx, byte ptr [r14 + rcx]
lfence
paddq xmm5, xmm6
lfence
adc rax, rcx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovnle rbx, qword ptr [r14 + rdi]
lfence
xor rcx, rax
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmp rcx, qword ptr [r14 + rcx]
lfence
pextrq rcx, xmm3, 0
lfence
cmp rcx, rdi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rdx, qword ptr [r14 + rdi]
lfence
pand xmm0, xmm1
lfence
setnz dil
lfence
cmp rcx, rsi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovl rdi, qword ptr [r14 + rsi]
lfence
setl dil
lfence
dec rbx
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
