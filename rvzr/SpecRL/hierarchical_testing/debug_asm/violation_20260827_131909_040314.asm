.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx 
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx 
mov rcx, 1424 
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax] 
test rcx, rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rax] 
sub rcx, rcx 
mov rsi, 3392 
dec rcx 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi 
setnz cl 
setb dl 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
