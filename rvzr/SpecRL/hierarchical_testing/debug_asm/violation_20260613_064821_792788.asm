.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx 
sbb rbx, rcx 
xor rbx, rax 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4000 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdx 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rcx 
imul rcx, rdx 
imul rdx, rsi 
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi 
imul rax, rsi 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx 
and rdx, rax 
imul rdx, rcx 
xor rdx, rcx 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rbx] 
xor rsi, rcx 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdi 
test rsi, rax 
setz bl 
dec rax 
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi 
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx] 
cmp rbx, rax 
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rcx 
sbb rbx, rax 
cmp rbx, rdx 
and rbx, rdi 
dec rsi 
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
