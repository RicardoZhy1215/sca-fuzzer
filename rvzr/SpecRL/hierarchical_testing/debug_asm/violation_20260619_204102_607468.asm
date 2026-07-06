.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rcx] 
psubq xmm4, xmm7 
por xmm4, xmm7 
setz al 
paddq xmm4, xmm5 
adc rsi, rdx 
pxor xmm4, xmm5 
and rsi, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx 
por xmm2, xmm1 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
setnz sil 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
not rdx 
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
imul rdi, rdi 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
or rdi, rbx 
or rsi, rbx 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
paddq xmm4, xmm5 
jmp .bb_0.1 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx] 
dec rsi 
setz sil 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax 
psubq xmm4, xmm7 
and rsi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx 
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rbx] 
setb sil 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1896 
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5536 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
