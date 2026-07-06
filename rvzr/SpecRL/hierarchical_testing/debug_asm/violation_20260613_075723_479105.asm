.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx 
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rax 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax] 
dec rax 
and rcx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
lea rbx, qword ptr [rdi + rbx + 1] 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi 
and rcx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi] 
xor rdx, rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx 
neg rbx 
and rcx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rsi 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
dec rdi 
and rcx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rcx] 
setl sil 
lea rdx, qword ptr [rsi + rdx + 1] 
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi 
and rcx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rcx] 
sbb rbx, rdx 
neg rsi 
inc rsi 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi 
inc rsi 
sbb rcx, rsi 
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rax 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rsi 
sub rsi, rax 
dec rsi 
xor rcx, rdx 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx 
not rbx 
and rcx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rdi 
dec rsi 
neg rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
