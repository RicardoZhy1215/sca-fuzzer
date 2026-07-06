.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5656 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3072 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2520 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7776 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6288 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4056 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 776 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4144 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3072 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1896 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6104 
neg rdx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7864 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5568 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6176 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3072 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 464 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7896 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5368 
and rdi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rcx 
and rdi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdi] 
neg rax 
and rdi, rdi 
xor rbx, rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7472 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4912 
and rdi, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
xor rcx, rsi 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rdi 
dec rsi 
inc rdi 
and rcx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rcx] 
inc rdx 
not rcx 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx 
and rdi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
