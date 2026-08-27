.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdx 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx] 
xor rbx, rsi 
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rdi] 
inc rbx 
setnz al 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1536 
and rdi, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rdx 
setb cl 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
inc rsi 
and rdx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi] 
setl cl 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
lea rdx, qword ptr [rsi + rdx + 1] 
and rbx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rsi 
mov rcx, 4888 
setnz cl 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi 
and rbx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rbx] 
cmp rdx, rax 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2208 
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx] 
setnz cl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
