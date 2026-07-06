.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi 
inc rdx 
not rax 
dec rbx 
adc rax, rcx 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
cmp rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi 
sub rbx, rdx 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdi 
neg rax 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
neg rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi 
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rbx 
mov rdx, 8168 
and rdi, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdx] 
mov rdx, rax 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rsi 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
sbb rdi, rbx 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
adc rax, rdx 
and rdi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rdx 
setz sil 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4 
and rdi, rax 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rax] 
imul rbx, rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4952 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdi 
lea rdx, qword ptr [rdi + rdx + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
