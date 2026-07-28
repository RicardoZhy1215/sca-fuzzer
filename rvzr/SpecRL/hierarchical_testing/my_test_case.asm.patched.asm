.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pmuludq xmm0, xmm7
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rcx
mov rsi, 4456
and rdi, rax
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi
and rax, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rax]
not rsi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
psubq xmm4, xmm4
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rdi
lea rbx, qword ptr [rcx + rbx + 1]
adc rcx, rsi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5912
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx]
adc rdi, rdx
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rdi
setb sil
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdi
setb bl
not rdi
and rdi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdi]
mov rax, 4848
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx]
paddq xmm5, xmm6
adc rax, rcx
and rdi, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdi]
xor rcx, rax
and rcx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rcx]
pextrq rcx, xmm3, 0
cmp rcx, rdi
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rdi]
pand xmm0, xmm1
setnz dil
cmp rcx, rsi
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
setl dil
dec rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
