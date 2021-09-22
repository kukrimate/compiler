// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use super::ast;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::Command;
use std::rc::Rc;
use tempfile::tempdir;

// Generate assembly code for a static initializer
fn gen_static_init(file: &ast::File, dtype: &Rc<ast::Type>, init: &ast::Init, output: &mut fs::File) {
    match &**dtype {
        ast::Type::Array { elem_type, elem_count } => {
            let init_list = init.want_list();
            if init_list.len() != *elem_count {
                panic!("Invalid static array initializer");
            }
            for elem_init in init_list {
                gen_static_init(file, elem_type, elem_init, output);
            }
        },

        ast::Type::Record(record) => {
            // FIXME: this doesn't work for unions
            for ((_, (field_type, _)), field_init) in
                                record.fields.iter().zip(init.want_list()) {
                gen_static_init(file, field_type, field_init, output);
            }
        },

        _ => {
            match init.want_expr() {
                // Static can be initialized by a constant
                ast::Expr::Const(ctype, val) => {
                    if dtype != ctype {
                        panic!("Type mismatch for static initializer")
                    }

                    match &**dtype {
                        ast::Type::U8 | ast::Type::I8
                            => writeln!(output, "db 0x{:X}", *val as u8).unwrap(),
                        ast::Type::U16 | ast::Type::I16
                            => writeln!(output, "dw 0x{:X}", *val as u16).unwrap(),
                        ast::Type::U32 | ast::Type::I32
                            => writeln!(output, "dd 0x{:X}", *val as u32).unwrap(),
                        ast::Type::U64 | ast::Type::I64 | ast::Type::Ptr {..}
                            => writeln!(output, "dq 0x{:X}", *val as u64).unwrap(),

                        _ => panic!("Invalid type {:?} for constant", dtype)
                    }
                }

                // Or by the address of another static
                ast::Expr::Ref(expr) if let ast::Expr::Ident(ident) = &**expr
                    => match &**dtype {
                        // FIXME: check base type of pointer too
                        ast::Type::Ptr {..} => writeln!(output, "dq {}", ident).unwrap(),
                        _ => panic!("Non-pointer static initialized with pointer"),
                    },

                _ => panic!("Non constant static initializer")
            }
        }
    }
}

/*

// Registers usable as temporary values
// Rip, Rsp, Rbp are reserved for the compiler's use
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reg {
    Rax = (1 << 0),
    Rbx = (1 << 1),
    Rcx = (1 << 2),
    Rdx = (1 << 3),
    Rsi = (1 << 4),
    Rdi = (1 << 5),
    R8  = (1 << 6),
    R9  = (1 << 7),
    R10 = (1 << 8),
    R11 = (1 << 9),
    R12 = (1 << 10),
    R13 = (1 << 11),
    R14 = (1 << 12),
    R15 = (1 << 13),
}

impl Reg {
    fn to_str(&self, dtype: &Type) -> &str {
        let width = match dtype {
            Type::U8  => 0,
            Type::I8  => 0,
            Type::U16 => 1,
            Type::I16 => 1,
            Type::U32 => 2,
            Type::I32 => 2,
            Type::U64 => 3,
            Type::I64 => 3,
            Type::Ptr {..} => 3,
            _ => panic!("Invalid data type for rvalue"),
        };

        match self {
            Reg::Rax => ["al", "ax", "eax", "rax"][width],
            Reg::Rbx => ["bl", "bx", "ebx", "rbx"][width],
            Reg::Rcx => ["cl", "cx", "ecx", "rcx"][width],
            Reg::Rdx => ["dl", "dx", "edx", "rdx"][width],
            Reg::Rsi => ["sil", "si", "esi", "rsi"][width],
            Reg::Rdi => ["dil", "di", "edi", "rdi"][width],
            Reg::R8  => ["r8b", "r8w", "r8d", "r8"][width],
            Reg::R9  => ["r9b", "r9w", "r9d", "r9"][width],
            Reg::R10 => ["r10b", "r10w", "r10d", "r10"][width],
            Reg::R11 => ["r11b", "r11w", "r11d", "r11"][width],
            Reg::R12 => ["r12b", "r12w", "r12d", "r12"][width],
            Reg::R13 => ["r13b", "r13w", "r13d", "r13"][width],
            Reg::R14 => ["r14b", "r14w", "r14d", "r14"][width],
            Reg::R15 => ["r15b", "r15w", "r15d", "r15"][width],
        }
    }
}

// List of all registers (in allocation order, callee-saved ones come first)
static ALL_REGS: [Reg; 14] = [
    Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15, // Callee saved
    Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi, // Caller saved
    Reg::R8,  Reg::R9,  Reg::R10, Reg::R11,
];

// List of caller saved registers
static CALLER_SAVED: [Reg; 9] = [
    Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi,
    Reg::R8,  Reg::R9,  Reg::R10, Reg::R11,
];

// List of registers used for parameters
static PARAM_REGS: [Reg; 6] = [
    Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9
];

struct FuncCtx {
    // Allocated registers
    usedregs: u16,
    // Output assembly
    asmcode: String,
}

impl FuncCtx {
    fn alloc_reg(&mut self) -> Option<Reg> {
        for reg in ALL_REGS {
            let mask = reg as u16;
            if self.usedregs & mask == 0 {
                self.usedregs |= mask;
                return Some(reg);
            }
        }
        None
    }

    fn maybe_alloc_specific_reg(&mut self, want: Reg) -> Option<Reg> {
        let wantmask = want as u16;
        if self.usedregs & wantmask == 0 {
            self.usedregs |= wantmask;
            Some(want)
        } else {
            self.alloc_reg()
        }
    }

    fn really_alloc_specific_reg(&mut self, want: Reg) -> Option<Reg> {
        let wantmask = want as u16;
        if self.usedregs & wantmask == 0 {
            self.usedregs |= wantmask;
            Some(want)
        } else {
            None
        }
    }

    fn clear_reg(&mut self, reg: Reg) {
        let mask = reg as u16;
        if self.usedregs & mask == 0 {
            panic!("Register {:?} is already clear", reg);
        }
        self.usedregs &= !mask;
    }

    // Save all caller-saved registers before a call (except the one used for the return value)
    fn pushcallersaved(&mut self, ret: Reg) -> (u16, bool) {
        let mut savemask = 0u16;
        let mut pad = false;

        // Save all caller-saved regsiters marked as used
        for reg in CALLER_SAVED {
            let regmask = reg as u16;
            if reg != ret && self.usedregs & regmask != 0 {
                savemask |= regmask;
                pad = !pad;
                self.addasm(format!("push {}", reg.to_str(&Type::U64)));
            }
        }
        // Make sure stack alingment is preserved
        if pad {
            self.addasm(format!("push 0"));
        }
        // Clear the caller-saved registers in the use mask
        self.usedregs &= !savemask;

        (savemask, pad)
    }

    // Restore saved caller-saved registers
    fn popcallersaved(&mut self, savemask: u16, pad: bool) {
        // Remove padding if present
        if pad {
            self.addasm(format!("add rsp, 8"));
        }

        // Make sure we didn't acccidently clobber one of the saved registers
        if self.usedregs & savemask != 0 {
            panic!("Saved register clobbered in call");
        }

        // Restore all saved registers
        for reg in CALLER_SAVED.iter().rev() {
            let regmask = *reg as u16;
            if savemask & regmask != 0 {
                self.addasm(format!("pop {}", reg.to_str(&Type::U64)));
            }
        }

        // Set restored registers to clobbered state again
        self.usedregs |= savemask;
    }

    // Append a line of assembly code to the output
    fn addasm(&mut self, code: String) {
        self.asmcode.extend(code.chars());
        self.asmcode.push('\n');
    }
}

fn gen_func<T: std::io::Write>(file: &File, func: &Func, output: &mut T) {
    print_or_die!(output, "{}:", func.name);

    // Create context
    let mut ctx = FuncCtx {
        usedregs: 0,
        asmcode: String::new(),
    };

    // Allocate stack slots for parameters
    for (name, dtype) in &func.params {
        if dtype.get_size() > 8 {
            todo!("larger-than register parameters");
        }
        // Add local variable for parameter
        ctx.locals.insert(name.clone(), (dtype.clone(), ctx.frame_size));
        // Increase stack frame size
        // HACK!: assume all parameters are 8 bytes for simpler moves
        ctx.frame_size += 8;
    }

    // Allocate stack slots for auto variables
    for stmt in &func.stmts {
        match stmt {
            Stmt::Auto(name, dtype, _) => {
                ctx.locals.insert(name.clone(), (dtype.clone(), ctx.frame_size));
                ctx.frame_size += dtype.get_size();
            },
            _ => (),
        }
    }

    // Align stack frame size to a multiple of 16 + 8 (SysV ABI requirement)
    ctx.frame_size = (ctx.frame_size + 15) / 16 * 16 + 8;
    // Generate function prologue
    print_or_die!(output, "push rbp\nmov rbp, rsp\nsub rsp, {}", ctx.frame_size);

    // Generate function epilogue
    print_or_die!(output, ".$done:\nleave\nret");
}

*/

pub fn gen_asm(input: &ast::File, output: &mut fs::File) {
    let mut exports = Vec::new();
    let mut externs = Vec::new();

    let mut bss = HashMap::new();

    // Generate data
    writeln!(output, "section .data").unwrap();
    for (_, cur_static) in &input.statics {
        match cur_static.vis {
            ast::Vis::Export => exports.push(cur_static.name.clone()),
            ast::Vis::Extern => externs.push(cur_static.name.clone()),
            _ => (),
        };

        match &cur_static.init {
            // Generate static initializer in .data
            Some(init) => {
                writeln!(output, "{}:", cur_static.name).unwrap();
                gen_static_init(&input, &cur_static.dtype, init, output);
            },
            // Take note for bss allocation later
            None => {
                bss.insert(cur_static.name.clone(),
                    cur_static.dtype.get_size());
            },
        };
    }

    // Generate bss
    writeln!(output, "section .bss").unwrap();
    for (name, len) in bss {
        writeln!(output, "{}: resb {}", name, len).unwrap();
    }

    // Generate functions
    // print_or_die!(output, "section .text");

    // for (_, func) in &input.funcs {
    //     match func.vis {
    //         Vis::Export => exports.push(func.name.clone()),
    //         Vis::Extern => externs.push(func.name.clone()),
    //         _ => (),
    //     };
    //     // Generate body for non-extern function
    //     if func.vis != Vis::Extern {
    //         gen_func(&input, &func, output);
    //     }
    // }

    // Generate markers
    for exp in exports {
        writeln!(output, "global {}", exp).unwrap();
    }
    for ext in externs {
        writeln!(output, "extern {}", ext).unwrap();
    }
}

pub fn gen_obj(input: &ast::File, output: &mut fs::File) {
    let tdir = tempdir().unwrap();
    let tasm = tdir.path().join("asm");
    let tobj = tdir.path().join("obj");

    // Generate assembly
    gen_asm(input, &mut fs::File::create(&tasm).unwrap());

    // Assemble
    let status = Command::new("nasm")
        .args(["-f", "elf64", "-o", tobj.to_str().unwrap(),
                tasm.to_str().unwrap()])
        .status()
        .expect("failed to run nasm");

    if !status.success() {
        panic!("assembly with nasm failed");
    }

    // Write assembly result to output
    output.write(&std::fs::read(tobj).unwrap()).unwrap();
}
