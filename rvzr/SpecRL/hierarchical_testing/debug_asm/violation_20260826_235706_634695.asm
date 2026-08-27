.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx 
mov rdx, 3968 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rcx 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdx] 
inc rcx 
sub rdi, rax 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rax] 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3728 
and rbx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rbx] 
dec rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx 
and rsi, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rsi] 
lea rbx, qword ptr [rcx + rbx + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
