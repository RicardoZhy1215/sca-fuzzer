.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
setl dil 
lea rsi, qword ptr [rsi + rsi + 1] 
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi] 
dec rcx 
lea rsi, qword ptr [rdx + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2304 
setnz cl 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx] 
test rbx, rbx 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdx 
and rsi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
