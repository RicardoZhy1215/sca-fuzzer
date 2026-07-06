.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rsi, rdx 
and rdi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx 
or rcx, rcx 
mov rsi, 4376 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdx 
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
pxor xmm0, xmm3 
and rcx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rcx] 
setb sil 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2880 
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
setnz dl 
setb sil 
dec rsi 
setb dil 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1544 
lea rsi, qword ptr [rdx + rsi + 1] 
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rcx] 
setb bl 
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi 
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi] 
pxor xmm4, xmm5 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
adc rdi, rbx 
test rsi, rbx 
psubq xmm4, xmm0 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
pxor xmm4, xmm5 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
psubq xmm4, xmm1 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
