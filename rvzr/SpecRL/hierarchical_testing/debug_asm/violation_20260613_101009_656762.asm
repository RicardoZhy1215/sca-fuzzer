.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
adc rdx, rsi 
and rdx, rcx 
cmp rbx, rax 
cmp rdi, rsi 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
and rbx, rax 
and rsi, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rsi] 
setnz dl 
sub rax, rax 
neg rdx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6448 
sub rdx, rdi 
cmp rdi, rax 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rsi] 
xor rax, rcx 
and rsi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rsi] 
test rdi, rcx 
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi 
sub rsi, rbx 
cmp rdx, rdi 
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
