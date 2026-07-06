.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, rbx 
setz dl 
setnz dl 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
imul rdx, rdi 
and rbx, rax 
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rax] 
adc rdx, rbx 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi 
and rax, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rcx] 
setz dl 
setnz sil 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi 
adc rbx, rax 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rax 
setz dil 
not rdx 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx 
and rsi, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdx] 
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx 
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rbx 
inc rsi 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rsi 
neg rdx 
and rdx, rdi 
neg rsi 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 376 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
cmp rdx, rdi 
and rdi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdi] 
mov rdx, 3896 
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rsi 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx 
and rdi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdi] 
inc rdx 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rsi 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rsi 
and rax, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdx] 
setl bl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
