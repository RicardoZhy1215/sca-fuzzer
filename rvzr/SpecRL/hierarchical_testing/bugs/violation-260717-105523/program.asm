.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 46 # instrumentation
imul rdi, rdx
and rdx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rax
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax
imul rdi, rdi
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rsi
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rcx, rax
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
setnz cl
and rcx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rsi
jbe .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdx]
imul rsi, rax
imul rdi, rax
setnz dil
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx
imul rcx, rdi
imul rsi, rdi
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi]
inc rdi
mov rsi, 4416
and rdx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdi]
imul rdi, rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdx
neg rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
