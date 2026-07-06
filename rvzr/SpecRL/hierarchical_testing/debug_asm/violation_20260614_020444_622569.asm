.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4144 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4144 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2792 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4040 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 16 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4144 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7008 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3072 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4144 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3864 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 6136 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2520 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4144 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4144 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 344 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3480 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5808 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4456 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5992 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1760 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5928 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rdi 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4144 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rcx 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7616 
xor rdx, rdi 
and rdi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
neg rdx 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdi 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rcx 
and rbx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdi 
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5384 
inc rdi 
xor rdi, rcx 
inc rdx 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
