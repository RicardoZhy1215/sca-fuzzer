.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, 84 # instrumentation
or rbx, rax
neg rax
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
cmp rdi, rdi
adc rdi, rdx
inc rax
and rbx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi
and rax, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdi]
mov rdi, 2376
setl al
or rcx, rdx
mov rax, 7272
or rcx, rsi
xor rsi, rdi
lea rax, qword ptr [rcx + rax + 1]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx
and rax, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
jnb .bb_0.1
jmp .exit_0
.bb_0.1:
setz bl
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rsi
cmp rsi, rbx
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
or rdi, rax
and rdi, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdi
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdx
adc rax, rbx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
lea rax, qword ptr [rdx + rax + 1]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
mov rax, rdx
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
