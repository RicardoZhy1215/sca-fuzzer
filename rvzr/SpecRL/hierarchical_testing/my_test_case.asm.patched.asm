.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
sbb rdi, rsi
and rsi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rsi]
sub rbx, rsi
sub rax, rdi
setnz cl
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4344
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rbx
setl bl
setz dil
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdi]
adc rdx, rdi
and rdi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rax]
adc rcx, rax
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
test rax, rcx
and rax, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rax]
adc rbx, rax
mov rdx, rcx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
