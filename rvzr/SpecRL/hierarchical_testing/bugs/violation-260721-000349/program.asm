.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -123 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rsi
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rcx
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rsi
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rsi
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
setl cl
and rcx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rcx]
setb dl
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx]
setl al
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi]
setl dl
and rdi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rdi]
imul rsi, rdx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx
and rsi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rsi
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rax]
imul rsi, rcx
and rdx, rbx
adc rax, rsi
imul rdx, rcx
jnb .bb_0.1
jmp .exit_0
.bb_0.1:
mov rdx, rbx
xor rdi, rcx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2984
and rdx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdx]
lea rdx, qword ptr [rdx + rdx + 1]
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx]
not rcx
and rax, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rax]
setb dl
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2208
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
mov rcx, 4952
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
neg rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
