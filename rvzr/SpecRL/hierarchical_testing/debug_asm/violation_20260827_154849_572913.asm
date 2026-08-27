.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
setz cl 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax 
setb bl 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rcx 
cmp rcx, rcx 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rdx] 
setz dl 
cmp rbx, rdi 
and rcx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
mov rdi, 1424 
dec rdx 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rsi 
setz al 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rbx] 
setnz bl 
mov rax, 3480 
mov rdi, 6640 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi 
and rax, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx] 
setnz cl 
imul rdi, rsi 
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx] 
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx 
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
