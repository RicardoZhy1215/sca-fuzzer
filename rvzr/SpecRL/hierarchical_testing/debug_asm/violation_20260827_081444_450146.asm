.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx] 
test rdi, rdx 
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rdx] 
inc rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
test rbx, rsi 
xor rdx, rsi 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
setl dl 
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
