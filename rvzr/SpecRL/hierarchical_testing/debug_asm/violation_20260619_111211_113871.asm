.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rdi 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
setnz sil 
adc rdi, rbx 
mov rsi, 6080 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx] 
paddq xmm5, xmm4 
and rdx, rsi 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
pand xmm2, xmm2 
setnz al 
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
setb al 
jmp .bb_0.1 
.bb_0.1:
cmp rsi, rdi 
not rsi 
mov rax, rbx 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdx 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx 
imul rax, rbx 
mov rax, rsi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3296 
and rcx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi] 
pxor xmm1, xmm1 
and rsi, rax 
paddq xmm6, xmm4 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
not rax 
pcmpeqd xmm3, xmm5 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi 
and rbx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rbx] 
setb sil 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
setb dl 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
