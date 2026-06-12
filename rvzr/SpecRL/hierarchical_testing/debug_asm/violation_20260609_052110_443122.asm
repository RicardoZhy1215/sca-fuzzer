.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
sbb rax, rsi 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
sub rcx, rsi 
and rbx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4936 
mov rax, rdi 
mov rax, rsi 
lea rdi, qword ptr [rax + rdi + 1] 
mov rax, rbx 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rbx 
xor rcx, rbx 
xor rdi, rbx 
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rsi 
sub rbx, rsi 
and rbx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
sub rax, rbx 
sbb rax, rdi 
mov rax, 1784 
sbb rax, rsi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe scasb  
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
mov rdx, 568 
and rdx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdx] 
mov rdx, 5832 
mov rax, 1160 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rax] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe stosb  
xor rax, rsi 
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax] 
lea rdx, qword ptr [rbx + rdx + 1] 
mov rbx, 6920 
mov rax, 7520 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
