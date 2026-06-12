.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rax, 1 # instrumentation
lfence
and rdx, rax # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rax
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add rax, qword ptr [r14 + rbx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmp rax, qword ptr [r14 + rbx]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rdi], rcx
lfence
and rdx, rsi
lfence
setz dl
lfence
imul rax, rdi
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
add rbx, qword ptr [r14 + rdx]
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov rdx, qword ptr [r14 + rdx]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mov rsi, qword ptr [r14 + rcx]
lfence
mov rdi, 1456
lfence
setb bl
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rax], rbx
lfence
setz al
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock xadd qword ptr [r14 + rsi], rdx
lfence
neg rdx
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rbx], rcx
lfence
and rbx, rdi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovle rsi, qword ptr [r14 + rdi]
lfence
not rbx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov rsi, qword ptr [r14 + rdi]
lfence
setnz sil
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdx], 1624
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmovnle rdx, qword ptr [r14 + rdx]
lfence
xor rsi, rdi
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mov rbx, qword ptr [r14 + rcx]
lfence
and rbx, rbx
lfence
setnz al
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
or rax, 1 # instrumentation
lfence
cmovnz rax, qword ptr [r14 + rsi]
lfence
and rdx, 0b1111111111111 # instrumentation
lfence
cmovns rax, qword ptr [r14 + rdx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rbx], rdx
lfence
and rcx, rsi
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rbx]
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rbx], rdi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], 5496
lfence
setz dil
lfence
dec rbx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovnle rbx, qword ptr [r14 + rsi]
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
