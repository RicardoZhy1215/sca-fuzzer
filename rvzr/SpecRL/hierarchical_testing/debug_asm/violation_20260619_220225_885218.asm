.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdi 
mov rax, 5856 
sbb rbx, rax 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx 
pxor xmm4, xmm5 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
inc rbx 
psubq xmm2, xmm5 
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax] 
sbb rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax] 
setnz dil 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
setl al 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx] 
inc rcx 
setnz dl 
jmp .bb_0.1 
.bb_0.1:
setb al 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
setb sil 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdx 
setb sil 
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx 
dec rbx 
and rbx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rbx] 
setb dil 
setb sil 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
setnz al 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx] 
setnz al 
setl dil 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
