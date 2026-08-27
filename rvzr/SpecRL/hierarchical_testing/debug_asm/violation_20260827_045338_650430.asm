.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
setl cl 
or rsi, rsi 
and rsi, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
not rdx 
sub rcx, rax 
adc rax, rsi 
setl bl 
lea rdi, qword ptr [rsi + rdi + 1] 
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax] 
cmp rax, rax 
lea rbx, qword ptr [rdx + rbx + 1] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
