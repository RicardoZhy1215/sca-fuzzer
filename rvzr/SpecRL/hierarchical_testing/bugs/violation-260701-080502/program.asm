.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
neg rsi
not rdi
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rcx
not rsi
sbb rax, rdi
and rdi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
mov rax, 6008
setb bl
not rsi
xor rcx, rsi
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdi
setnz bl
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
setnz dl
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx]
imul rax, rdx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rbx
and rcx, rsi
sbb rdx, rdx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax]
jmp .bb_0.1
.bb_0.1:
or rsi, rax
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rsi
inc rdx
setnz al
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
mov rcx, rdx
and rdi, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rdi]
setl sil
adc rdx, rdi
mov rdx, rax
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
xor rax, rax
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx]
mov rcx, rdx
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rdx, rax
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
