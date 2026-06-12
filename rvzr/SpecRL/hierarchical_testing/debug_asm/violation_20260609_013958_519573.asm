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
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
