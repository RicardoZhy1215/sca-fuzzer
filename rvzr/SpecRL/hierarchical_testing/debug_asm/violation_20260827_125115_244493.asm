.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
inc rcx 
and rcx, rax 
not rcx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2720 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi 
and rdx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdx] 
setnz cl 
and rcx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi 
adc rax, rsi 
dec rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
or rcx, rax 
and rdx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rsi] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi 
and rax, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rax] 
mov rdx, 2976 
imul rsi, rsi 
and rcx, rdi 
setnz cl 
setl sil 
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rax 
test rsi, rsi 
sub rbx, rax 
mov rcx, 3904 
sub rcx, rdi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi 
test rdx, rbx 
setnz dil 
test rdx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
