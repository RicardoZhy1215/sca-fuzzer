.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rcx
and rdx, rsi
setz dl
imul rax, rdi
and rdx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx]
mov rdi, 1456
setb bl
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx
setz al
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdx
neg rdx
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rbx, rdi
and rdi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdi]
not rbx
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi]
setnz sil
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1624
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx]
xor rsi, rdi
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx]
and rbx, rbx
setnz al
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdx
and rcx, rsi
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5496
setz dil
dec rbx
and rsi, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
