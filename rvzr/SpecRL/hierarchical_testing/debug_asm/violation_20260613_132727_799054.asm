.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 632 
test rdx, rcx 
setz dl 
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rbx] 
not rdx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi 
or rdx, rax 
and rax, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi 
lea rdx, qword ptr [rdx + rdx + 1] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3216 
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax] 
neg rbx 
and rax, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rax] 
inc rsi 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rax 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
inc rdx 
dec rsi 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rax] 
sub rdx, rbx 
or rdx, rax 
dec rdx 
and rcx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rbx 
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx 
setz dil 
xor rsi, rbx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx] 
setz sil 
and rax, rbx 
dec rsi 
and rax, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rax] 
dec rsi 
lea rax, qword ptr [rbx + rax + 1] 
setz cl 
or rsi, rax 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdx 
inc rax 
setl bl 
inc rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
