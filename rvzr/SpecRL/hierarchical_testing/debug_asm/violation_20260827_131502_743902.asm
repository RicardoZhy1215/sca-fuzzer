.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rcx 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
lea rdx, qword ptr [rbx + rdx + 1] 
imul rcx, rbx 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
setnz cl 
imul rsi, rbx 
inc rbx 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
inc rdi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx 
and rcx, rbx 
and rax, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
and rdx, rbx 
and rsi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
