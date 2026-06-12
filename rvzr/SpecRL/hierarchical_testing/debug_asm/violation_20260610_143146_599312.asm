.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rcx, rsi 
xor rdi, rcx 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx] 
xor rdi, rcx 
sub rdi, rcx 
and rcx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rcx] 
xor rdi, rcx 
xor rdi, rcx 
xor rdi, rcx 
sub rbx, rcx 
xor rbx, rcx 
xor rbx, rcx 
xor rbx, rcx 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rcx] 
xor rdi, rcx 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx 
xor rbx, rbx 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdi] 
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx 
xor rdi, rcx 
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdi 
lea rdi, qword ptr [rcx + rdi + 1] 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
sub rbx, rcx 
sub rcx, rbx 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rcx] 
xor rdi, rsi 
mov rsi, 456 
sub rdi, rcx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi 
sub rdi, rbx 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
sub rdx, rcx 
sub rdi, rcx 
mov rsi, 7816 
sub rax, rbx 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2192 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 776 
mov rsi, 4320 
xor rdi, rdi 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
xor rcx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
