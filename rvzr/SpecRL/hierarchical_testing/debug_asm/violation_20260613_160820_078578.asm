.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax 
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx] 
adc rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1120 
mov rdx, 7968 
inc rdx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2672 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2904 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6400 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1288 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2120 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6648 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3968 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4832 
test rdx, rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6744 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1576 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1536 
and rsi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3856 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3976 
neg rbx 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rax 
cmp rax, rsi 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
mov rax, rdi 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5872 
dec rsi 
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdi] 
dec rsi 
neg rsi 
and rsi, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rsi] 
and rsi, rdx 
and rdi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdi] 
and rsi, rcx 
inc rsi 
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
