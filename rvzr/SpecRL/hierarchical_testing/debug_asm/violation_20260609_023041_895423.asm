.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe scasb  
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rbx] 
mov rbx, rax 
mov rsi, 5592 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
mov rdx, 32 
sub rax, rsi 
sub rax, rsi 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rdi 
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rsi 
mov rsi, rax 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe cmpsb  
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe scasb  
mov rax, 4536 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe cmpsb  
lea rax, qword ptr [rbx + rax + 1] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe cmpsb  
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rax 
and rbx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rbx] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe scasb  
mov rax, 2088 
and rsi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rsi] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe scasb  
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
