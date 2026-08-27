.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rdi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 936 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 272 
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx] 
setl dl 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx 
setb bl 
sub rax, rbx 
lea rdi, qword ptr [rax + rdi + 1] 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rsi 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
test rsi, rbx 
test rcx, rbx 
setnz cl 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi 
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rbx 
imul rdi, rdx 
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
lea rcx, qword ptr [rsi + rcx + 1] 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rsi, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rax 
lea rbx, qword ptr [rdx + rbx + 1] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rdx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdx] 
imul rax, rax 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
lea rcx, qword ptr [rsi + rcx + 1] 
and rdx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4936 
mov rcx, 6384 
dec rcx 
lea rdx, qword ptr [rsi + rdx + 1] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rdx 
test rbx, rbx 
mov rcx, 72 
xor rax, rsi 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi 
and rbx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
