.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdi 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rax 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe stosb  
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx] 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
mov rcx, 5352 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rsi] 
sub rax, rdi 
lea rdi, qword ptr [rbx + rdi + 1] 
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5968 
lea rcx, qword ptr [rcx + rcx + 1] 
sbb rax, rsi 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdx 
sbb rbx, rdx 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe stosb  
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax] 
sub rsi, rbx 
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi] 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
mov rbx, rsi 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe stosb  
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
mov rax, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
