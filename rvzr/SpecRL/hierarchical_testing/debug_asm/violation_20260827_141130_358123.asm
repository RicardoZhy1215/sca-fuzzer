.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
adc rbx, rsi 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi 
test rbx, rbx 
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx] 
lea rcx, qword ptr [rbx + rcx + 1] 
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax 
and rdx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rcx] 
test rbx, rbx 
and rdx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
