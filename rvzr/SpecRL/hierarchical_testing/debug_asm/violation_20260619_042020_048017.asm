.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi] 
mov rsi, 5248 
and rcx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rsi] 
psubq xmm5, xmm3 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdx 
pmuludq xmm7, xmm0 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax 
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi] 
lea rdi, qword ptr [rax + rdi + 1] 
and rax, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rsi 
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdi 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi 
lea rdi, qword ptr [rbx + rdi + 1] 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rbx 
setb sil 
and rdi, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rdi] 
sub rax, rbx 
or rsi, rax 
and rbx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi] 
sub rdx, rcx 
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx] 
or rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3824 
sub rcx, rsi 
pmuludq xmm7, xmm5 
lea rdi, qword ptr [rdi + rdi + 1] 
setz cl 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rbx 
neg rdi 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rcx 
mov rdi, 6576 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rsi 
sub rbx, rbx 
setnz sil 
setb sil 
pxor xmm3, xmm1 
inc rdi 
sub rsi, rax 
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
and rdi, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdi] 
adc rbx, rsi 
jmp .bb_0.1 
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
