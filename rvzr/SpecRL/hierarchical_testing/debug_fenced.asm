.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
lfence
or rax, 1 # instrumentation
lfence
cmovnz rax, qword ptr [r14 + rcx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
or rsi, 1 # instrumentation
lfence
cmovnz rsi, qword ptr [r14 + rsi]
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rcx], rsi
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov rax, qword ptr [r14 + rbx]
lfence
setl dil
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
or rcx, 1 # instrumentation
lfence
cmovnz rcx, qword ptr [r14 + rdi]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add rbx, qword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock cmpxchg qword ptr [r14 + rbx], rbx
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rdx], rsi
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmp rax, qword ptr [r14 + rcx]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rax], rax
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], 6856
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rbx, qword ptr [r14 + rbx]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
or rcx, 1 # instrumentation
lfence
cmovnz rcx, qword ptr [r14 + rax]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovnle rdi, qword ptr [r14 + rdi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovnle rbx, qword ptr [r14 + rdi]
lfence
and rcx, 0b1111111111000 # instrumentation
lfence
lock xadd qword ptr [r14 + rcx], rcx
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovs rdx, qword ptr [r14 + rcx]
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdx], rcx
lfence
setz al
lfence
mov rax, rcx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmovbe rsi, qword ptr [r14 + rdx]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rdi], rsi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovns rsi, qword ptr [r14 + rdi]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rdi], rdi
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmp rcx, rcx # instrumentation
lfence
cmovz rcx, qword ptr [r14 + rcx]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rax]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovns rax, qword ptr [r14 + rcx]
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rdx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovnl rbx, qword ptr [r14 + rsi]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rax], rcx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovs rsi, qword ptr [r14 + rdi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovs rsi, qword ptr [r14 + rdi]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
cmovnle rdx, qword ptr [r14 + rax]
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rbx], rcx
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
add rax, qword ptr [r14 + rdx]
lfence
setb bl
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rcx], rdi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmp qword ptr [r14 + rdi], rbx
lfence
adc rbx, rbx
lfence
imul rbx, rdx
lfence
dec rdx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovbe rsi, qword ptr [r14 + rsi]
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rdx], rbx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovnle rsi, qword ptr [r14 + rsi]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
or rax, 1 # instrumentation
lfence
cmovnz rax, qword ptr [r14 + rcx]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovnle rsi, qword ptr [r14 + rcx]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmp rax, qword ptr [r14 + rdi]
lfence
not rax
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
or rdx, 1 # instrumentation
lfence
cmovnz rdx, qword ptr [r14 + rsi]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
