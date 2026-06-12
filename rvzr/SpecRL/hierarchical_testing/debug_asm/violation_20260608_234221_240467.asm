.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rcx, rdx 
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx 
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdi 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rbx 
and rdi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
sub rsi, rsi 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe stosb  
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
lea rbx, qword ptr [rsi + rbx + 1] 
lea rax, qword ptr [rbx + rax + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
