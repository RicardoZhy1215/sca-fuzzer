.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx] 
mov rdx, rsi 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx] 
and rcx, rdi 
and rdx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdx] 
mov rbx, rcx 
setz bl 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rdi 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdi 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax 
inc rax 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax 
mov rdi, 3944 
setl cl 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi 
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7432 
cmp rcx, rbx 
mov rdx, rcx 
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx] 
xor rdi, rcx 
sbb rsi, rcx 
setz bl 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rcx 
and rdx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdx] 
mov rbx, 7192 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
