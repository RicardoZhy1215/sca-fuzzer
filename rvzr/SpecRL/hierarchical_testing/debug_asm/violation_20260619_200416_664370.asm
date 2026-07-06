.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
setb al 
psubq xmm4, xmm6 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx] 
setl al 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
psubq xmm4, xmm1 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx] 
sbb rax, rbx 
and rdx, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rdx] 
setnz cl 
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 704 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rbx 
cmp rax, rbx 
setb sil 
imul rdi, rcx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
lea rsi, qword ptr [rax + rsi + 1] 
mov rax, 8000 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rcx] 
sub rsi, rdi 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5440 
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
setb sil 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
setb sil 
and rsi, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3160 
setb al 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
