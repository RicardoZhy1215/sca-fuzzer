.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 3 # instrumentation
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rsi 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi 
setz cl 
and rsi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdi 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx 
and rdi, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx 
and rcx, rbx 
neg rcx 
sub rdx, rbx 
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi 
dec rbx 
lea rdi, qword ptr [rdx + rdi + 1] 
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi] 
setl dil 
and rbx, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax] 
jle .bb_0.1 
jmp .exit_0 
.bb_0.1:
sub rsi, rdx 
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi] 
xor rsi, rax 
setnz dl 
and rsi, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rbx 
setb bl 
neg rdi 
and rbx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax] 
and rcx, rdi 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rsi] 
or rsi, rsi 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rsi 
xor rcx, rbx 
and rsi, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rsi] 
lea rdx, qword ptr [rcx + rdx + 1] 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
and rdx, rbx 
dec rbx 
and rsi, rdi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
