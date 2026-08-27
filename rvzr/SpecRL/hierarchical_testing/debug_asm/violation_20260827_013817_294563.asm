.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax 
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax] 
setnz bl 
setnz cl 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx 
cmp rdi, rsi 
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rcx 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rcx 
not rax 
adc rdi, rax 
not rcx 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
