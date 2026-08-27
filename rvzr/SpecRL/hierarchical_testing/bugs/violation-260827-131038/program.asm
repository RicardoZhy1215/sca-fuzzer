.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6200
sub rcx, rcx
and rcx, rsi
inc rbx
setl dl
mov rax, 2152
setnz cl
xor rcx, rbx
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi
imul rsi, rsi
mov rdi, rdx
lea rdx, qword ptr [rsi + rdx + 1]
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rdi]
sub rax, rcx
and rdi, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi
setnz cl
not rbx
setnz al
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdi
imul rsi, rdx
setl cl
adc rcx, rbx
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 6408
and rdx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
