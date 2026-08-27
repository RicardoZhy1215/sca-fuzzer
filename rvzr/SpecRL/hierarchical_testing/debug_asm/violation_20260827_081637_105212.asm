.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
sub rdx, rax 
setl cl 
test rdx, rsi 
test rdi, rcx 
inc rax 
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rbx 
lea rbx, qword ptr [rdx + rbx + 1] 
and rdi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi 
test rdx, rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax 
and rax, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi] 
dec rbx 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
not rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5760 
test rcx, rsi 
setz dl 
test rcx, rcx 
setb dil 
and rdx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 8056 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
