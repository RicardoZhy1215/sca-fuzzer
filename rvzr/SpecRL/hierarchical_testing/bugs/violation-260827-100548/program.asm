.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setl bl
lea rcx, qword ptr [rdx + rcx + 1]
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi]
inc rdx
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
test rsi, rsi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7136
and rsi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi
lea rdx, qword ptr [rdx + rdx + 1]
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
setnz dil
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx]
adc rdx, rdx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rsi
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
test rsi, rsi
setz cl
setnz cl
not rax
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx]
test rbx, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
