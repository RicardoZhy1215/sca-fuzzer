.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rdx]
neg rcx
adc rdi, rbx
test rbx, rbx
setl al
setb dil
and rdx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rdx]
dec rdi
imul rdi, rsi
and rdx, rcx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdi
and rax, rsi
and rdx, rbx
imul rdx, rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
setl dl
and rdx, rcx
lea rsi, qword ptr [rdx + rsi + 1]
test rdi, rbx
and rsi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx
dec rcx
mov rdi, rsi
and rsi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
lea rcx, qword ptr [rsi + rcx + 1]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
test rax, rbx
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi]
mov rcx, rbx
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
