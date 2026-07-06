.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rsi] 
mov rsi, 3256 
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx] 
pand xmm2, xmm5 
and rdx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7408 
setl al 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
psubq xmm2, xmm4 
cmp rdi, rsi 
setb dl 
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx] 
xor rdx, rcx 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax] 
setb sil 
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx] 
and rdi, rbx 
and rdi, rbx 
setb sil 
dec rdx 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdi 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rax 
and rbx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3840 
jmp .bb_0.1 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
pmuludq xmm5, xmm1 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
pxor xmm0, xmm5 
dec rsi 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
or rax, rbx 
pmuludq xmm7, xmm4 
inc rsi 
lea rsi, qword ptr [rax + rsi + 1] 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx 
setb sil 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
setb sil 
pmuludq xmm4, xmm5 
and rbx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
not rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
