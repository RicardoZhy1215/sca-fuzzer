.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
and rdx, rax 
setnz dl 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
dec rcx 
or rdi, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
