.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
cmp rax, rsi 
inc rcx 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
lea rdx, qword ptr [rsi + rdx + 1] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rbx] 
and rdx, rsi 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rsi 
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi] 
setnz cl 
or rsi, rdx 
and rdi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
setnz cl 
lea rcx, qword ptr [rbx + rcx + 1] 
and rbx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rbx] 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx 
setnz cl 
and rbx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
sbb rbx, rbx 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rdx 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rbx 
mov rdx, rax 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx 
test rsi, rsi 
adc rbx, rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
