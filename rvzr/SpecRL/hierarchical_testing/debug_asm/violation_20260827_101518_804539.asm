.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi] 
not rcx 
and rsi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rsi] 
mov rdx, 6416 
setb dil 
lea rbx, qword ptr [rdx + rbx + 1] 
and rax, rcx 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi 
inc rdx 
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
setnz sil 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rcx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx] 
mov rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdx] 
setb bl 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
not rdi 
setb dl 
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi] 
setnz al 
test rsi, rbx 
mov rcx, 3080 
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rsi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
and rdx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rsi 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx 
setl bl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
