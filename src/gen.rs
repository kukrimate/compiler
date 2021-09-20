// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use super::ast::{Type,Expr,Init,Stmt,Vis,Func,File};
use std::collections::HashMap;
use std::rc::Rc;
use std::process::Command;
use tempfile::tempdir;

macro_rules! print_or_die {
    ($file:expr, $($args:expr),*) => {
        writeln!($file, $($args),*).unwrap()
    }
}

fn gen_static_init<T: std::io::Write>(dtype: &Type, init: &Init, file: &File, output: &mut T) {
    match dtype {
        Type::Array { elem_type, elem_count } => {
            let init_list = init.want_list();
            if init_list.len() != *elem_count {
                panic!("Invalid static array initializer");
            }
            for elem_init in init_list {
                gen_static_init(elem_type, elem_init, file, output);
            }
        },

        Type::Record(record) => {
            for ((name, (dtype, _)), init) in record.fields.iter().zip(init.want_list()) {
                gen_static_init(dtype, init, file, output);
            }
        },

        _ => {
            match init.want_expr() {
                // Static can be initialized by a constant
                Expr::Const(ctype, v) => {
                    if dtype != ctype {
                        panic!("Type mismatch for static initializer")
                    }

                    match dtype {
                        Type::U8  => { print_or_die!(output, "db 0x{:X}", *v as u8); },
                        Type::I8  => { print_or_die!(output, "db 0x{:X}", *v as u8); },
                        Type::U16 => { print_or_die!(output, "dw 0x{:X}", *v as u16); },
                        Type::I16 => { print_or_die!(output, "dw 0x{:X}", *v as u16); },
                        Type::U32 => { print_or_die!(output, "dd 0x{:X}", *v as u32); },
                        Type::I32 => { print_or_die!(output, "dd 0x{:X}", *v as u32); },
                        Type::U64 => { print_or_die!(output, "dq 0x{:X}", *v as u64); },
                        Type::I64 => { print_or_die!(output, "dq 0x{:X}", *v as u64); },
                        Type::Ptr {..} => { print_or_die!(output, "dq 0x{:X}", *v as u64); },
                        _ => panic!("Invalid type {:?} for constant", dtype)
                    }
                }

                // Static can be initialized by another static
                Expr::Ident(ident) => {
                    if let Some(s) = file.statics.get(ident) {
                        if let Some(s_init) = s.init.as_ref() {
                            gen_static_init(dtype, s_init, file, output);
                        } else {
                            todo!("Static initialized by non-initialized static")
                        }
                    } else {
                        panic!("Unknown identifier {:?}", ident)
                    }
                },

                // Or by a pointer to a static (or string literal)
                Expr::Ref(dest) => {
                    match &**dest {
                        Expr::Ident(ident) => {
                            print_or_die!(output, "dq {}", ident);
                        },
                        _ => panic!("Non constant static initializer"),
                    }
                },

                _ => panic!("Non constant static initializer")
            }
        }
    }
}

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

struct FuncCtx<'a> {
    // Reference to the containing file
    file: &'a File,
    // Allocated registers
    usedregs: u16,
    // Local variables
    locals: HashMap<Rc<str>, (Type, usize)>,
    // Stack frame size
    frame_size: usize,
    // Output assembly
    asmcode: String,
}

impl<'a> FuncCtx<'a> {
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

#[derive(Clone)]
enum LVal {
    // Value is stored on the stack
    Stack(Type, usize),
    // Value is stored in static storage
    Static(Type, Rc<str>, usize),
    // Dereference of a pointer
    Deref(Type, RVal, usize),
}

impl LVal {
    fn get_type(&self) -> &Type {
        match self {
            LVal::Stack(dtype, _) => dtype,
            LVal::Static(dtype, _, _) => dtype,
            LVal::Deref(dtype, _, _) => dtype,
        }
    }

    // Create an lvalue pointing to a struct field
    fn record_field(&self, ident: &Rc<str>) -> LVal {
        let record = match self.get_type() {
            Type::Record(record) => record,
            _ => panic!("Field access on non-record type"),
        };

        if let Some((dtype, offset)) = record.fields.get(ident) {
            match self {
                LVal::Stack(_, o) => LVal::Stack(dtype.clone(), o + offset),
                LVal::Static(_, name, o) => LVal::Static(dtype.clone(), name.clone(), o + offset),
                LVal::Deref(_, ptr, o) => LVal::Deref(dtype.clone(), ptr.clone(), o + offset),
            }
        } else {
            panic!("Field {:?} doesn't exist", ident)
        }
    }

    // Create an lvalue pointing to an array element
    fn array_element(&self, idx: RVal, fctx: &mut FuncCtx) -> LVal {
        let (elem_type, elem_count) = match self.get_type() {
            Type::Array { elem_type, elem_count } => (&**elem_type, *elem_count),
            _ => panic!("Element access on non-array type"),
        };

        if let RVal::Immed(_, val) = idx {
            if val >= elem_count {
                panic!("Const array index out of bounds!")
            }

            match self {
                LVal::Stack(_, o) => LVal::Stack(elem_type.clone(),
                                                o + val * elem_type.get_size()),
                LVal::Static(_, name, o) => LVal::Static(elem_type.clone(),
                                                name.clone(), o + val * elem_type.get_size()),
                LVal::Deref(_, ptr, o) => LVal::Deref(elem_type.clone(),
                                                ptr.clone(), o + val * elem_type.get_size()),
            }
        } else {
            todo!("non-constant array index")
        }
    }

    // Create an rvalue from this lvalue
    fn to_rval(self, fctx: &mut FuncCtx) -> RVal {
        match self {
            LVal::Stack(dtype, offset) => {
                let dreg = fctx.alloc_reg().unwrap();
                fctx.addasm(format!("mov {}, [rbp - {}]",
                                        dreg.to_str(&dtype),
                                        fctx.frame_size - offset));
                RVal::Reg(dtype, dreg)
            },
            LVal::Static(dtype, name, offset) => {
                let dreg = fctx.alloc_reg().unwrap();
                fctx.addasm(format!("mov {}, [{} + {}]",
                                        dreg.to_str(&dtype),
                                        name, offset));
                RVal::Reg(dtype, dreg)
            },
            LVal::Deref(dtype, ptr, offset) => {
                // Store pointer value in a register
                let dreg = ptr.to_reg(fctx);
                // Perform the dereference
                fctx.addasm(format!("mov {}, [{} + {}]",
                                        dreg.to_str(&dtype),
                                        dreg.to_str(&Type::U64), offset));
                RVal::Reg(dtype, dreg)
            },
        }
    }

    // Create an rvalue pointer to this lvalue
    fn to_ptr(self, fctx: &mut FuncCtx) -> RVal {
        match self {
            LVal::Stack(dtype, offset) => {
                let dreg = fctx.alloc_reg().unwrap();
                fctx.addasm(format!("lea {}, [rbp - {}]",
                                        dreg.to_str(&Type::U64),
                                        fctx.frame_size - offset));
                RVal::Reg(Type::Ptr { base_type: Box::from(dtype) }, dreg)
            },
            LVal::Static(dtype, name, offset) => {
                let dreg = fctx.alloc_reg().unwrap();
                fctx.addasm(format!("lea {}, [{} + {}]",
                                        dreg.to_str(&Type::U64),
                                        name, offset));
                RVal::Reg(Type::Ptr { base_type: Box::from(dtype) }, dreg)
            },
            LVal::Deref(dtype, ptr, offset) => {
                if offset > 0 {
                    // Add offset to pointer if required
                    let dreg = ptr.to_reg(fctx);
                    fctx.addasm(format!("lea {}, [{} + {}]",
                                    dreg.to_str(&Type::U64),
                                    dreg.to_str(&Type::U64), offset));
                    RVal::Reg(Type::Ptr { base_type: Box::from(dtype) }, dreg)
                } else {
                    ptr
                }
            },
        }
    }
}

#[derive(Clone)]
enum RVal {
    // Immediate integer value
    Immed(Type, usize),
    // Integer (or pointer) value in a register
    Reg(Type, Reg),
}

impl RVal {
    fn get_type(&self) -> &Type {
        match self {
            RVal::Immed(dtype, _) => dtype,
            RVal::Reg(dtype, _,) => dtype,
        }
    }

    fn to_reg(self, fctx: &mut FuncCtx) -> Reg {
        match self {
            RVal::Immed(dtype, val) => {
                let dreg = fctx.alloc_reg().unwrap();
                fctx.addasm(format!("mov {}, {}", dreg.to_str(&dtype), val));
                dreg
            },
            RVal::Reg(_, reg) => reg,
        }
    }

    fn to_string(&self) -> String {
        match self {
            RVal::Immed(_, val) => format!("{}", val),
            RVal::Reg(dtype, reg) => String::from(reg.to_str(&dtype)),
        }
    }
}

#[derive(Clone)]
enum Val {
    // Addressable value
    LVal(LVal),
    // Unaddressable temporary value
    RVal(RVal),
}

impl Val {
    fn discard(self, fctx: &mut FuncCtx) {
        match self {
            Val::RVal(rval) => match rval {
                RVal::Reg(_, reg) => fctx.clear_reg(reg),
                _ => (),
            },
            Val::LVal(lval) => match lval {
                LVal::Deref(_, ptr, _) => match ptr {
                    RVal::Reg(_, reg) => fctx.clear_reg(reg),
                    _ => (),
                }
                _ => (),
            },
        }
    }
}

fn gen_bop(fctx: &mut FuncCtx, op: &str, expr1: &Expr, expr2: &Expr) -> Val {
    let lhs = gen_rval_expr(fctx, expr1);
    let rhs = gen_rval_expr(fctx, expr2);
    if lhs.get_type() != rhs.get_type() {
        panic!("Type mismatch, left: {:?}, right: {:?}", lhs.get_type(), rhs.get_type());
    }
    if let RVal::Reg(dtype, reg) = &lhs {
        fctx.addasm(format!("{} {}, {}", op, reg.to_str(&dtype), rhs.to_string()));
        Val::RVal(lhs)
    } else if let RVal::Reg(dtype, reg) = &rhs {
        fctx.addasm(format!("{} {}, {}", op, reg.to_str(&dtype), lhs.to_string()));
        Val::RVal(rhs)
    } else {
        unreachable!();
    }
}

fn gen_expr(fctx: &mut FuncCtx, in_expr: &Expr) -> Val {

    match in_expr {
        Expr::Const(dtype, val) => Val::RVal(RVal::Immed(dtype.clone(), *val)),

        Expr::Ident(ident) => {
            if let Some((dtype, offset)) = fctx.locals.get(ident) {
                Val::LVal(LVal::Stack(dtype.clone(), *offset))
            } else if let Some(s) = fctx.file.statics.get(ident) {
                Val::LVal(LVal::Static(s.dtype.clone(), ident.clone(), 0))
            } else {
                panic!("Unknown identifier {}", ident);
            }
        },

        Expr::Ref(expr) =>
            Val::RVal(gen_lval_expr(fctx, expr).to_ptr(fctx)),

        Expr::Deref(expr) => {
            let ptr = gen_rval_expr(fctx, expr);
            if let Type::Ptr { base_type } = ptr.get_type() {
                Val::LVal(LVal::Deref((**base_type).clone(), ptr, 0))
            } else {
                panic!("Dereference of non-pointer type")
            }
        }

        Expr::Field(expr, ident) =>
            Val::LVal(gen_lval_expr(fctx, expr).record_field(ident)),

        Expr::Elem(expr, idx_expr) => {
            let arr = gen_lval_expr(fctx, expr);
            if let Type::Array { elem_type, elem_count } = arr.get_type() {
                let idx = gen_rval_expr(fctx, idx_expr);
                Val::LVal(arr.array_element(idx, fctx))
            } else {
                panic!("Element access on non-array type")
            }
        }

        Expr::Call(expr, params) => {
            // Find target function to call
            let func = match &**expr {
                Expr::Ident(ident) => {
                    if let Some(func) = fctx.file.funcs.get(ident) {
                        func
                    } else {
                        panic!("Call to undefined function {}", ident);
                    }
                },
                _ => panic!("Function name must be an identifier"),
            };

            // Allocate destination for the returned value
            // NOTE: we prefer to allocate rax here to avoid extra moves
            // FIXME: might not be able to allocate a register
            let dreg = fctx.maybe_alloc_specific_reg(Reg::Rax).unwrap();

            // Save caller-saved registers
            let (savemask, pad) = fctx.pushcallersaved(dreg);

            // Evaluate parameter expressions
            for (i, param) in params.iter().enumerate() {
                let param_reg = fctx.really_alloc_specific_reg(PARAM_REGS[i]).unwrap();
                match gen_rval_expr(fctx, param) {
                    RVal::Immed(dtype, val) => {
                        fctx.addasm(format!("mov {}, {}",
                            param_reg.to_str(&dtype), val));
                    },
                    RVal::Reg(dtype, reg) => {
                        if param_reg != reg {
                            fctx.addasm(format!("mov {}, {}",
                                param_reg.to_str(&dtype), reg.to_str(&dtype)));
                            fctx.clear_reg(reg);
                        }
                    },
                }
            }

            if func.varargs {
                // Provide number of floating arguments for varargs functions
                fctx.addasm(format!("xor rax, rax"));
            }
            // Call function
            fctx.addasm(format!("call {}", func.name));

            // Move return value to result register
            if dreg != Reg::Rax {
                fctx.addasm(format!("mov {}, rax", dreg.to_str(&func.rettype)));
                fctx.clear_reg(Reg::Rax);
            }
            // Clear param clobbers
            for (i, _) in params.iter().enumerate() {
                fctx.clear_reg(PARAM_REGS[i]);
            }

            // Restore caller-saved registers
            fctx.popcallersaved(savemask, pad);

            Val::RVal(RVal::Reg(func.rettype.clone(), dreg))
        },

        Expr::Inv(expr) => {
            let src = gen_rval_expr(fctx, expr);
            if let RVal::Reg(dtype, reg) = &src {
                fctx.addasm(format!("not {}", reg.to_str(&dtype)));
                Val::RVal(src)
            } else {
                unreachable!()
            }
        },

        Expr::Neg(expr) => {
            let src = gen_rval_expr(fctx, expr);
            if let RVal::Reg(dtype, reg) = &src {
                fctx.addasm(format!("neg {}", reg.to_str(&dtype)));
                Val::RVal(src)
            } else {
                unreachable!()
            }
        },

        Expr::Add(lhs, rhs) => gen_bop(fctx, "add", lhs, rhs),
        Expr::Sub(lhs, rhs) => gen_bop(fctx, "sub", lhs, rhs),

        Expr::Mul(lhs, rhs) => {
            todo!("multiply")
        },

        Expr::Div(lhs, rhs) => {
            todo!("divide")
        },

        Expr::Rem(lhs, rhs) => {
            todo!("remainder")
        },

        Expr::Or(lhs, rhs) => gen_bop(fctx, "or", lhs, rhs),
        Expr::And(lhs, rhs) => gen_bop(fctx, "and", lhs, rhs),
        Expr::Xor(lhs, rhs) => gen_bop(fctx, "xor", lhs, rhs),

        Expr::Lsh(lhs, rhs) => {
            todo!("left shift")
        },

        Expr::Rsh(lhs, rhs) => {
            todo!("right shift")
        },

        Expr::Cast(expr, dtype) => {
            todo!("cast")
            // let src = gen_expr(fctx, expr);
            // (dtype.clone(), src_val)
        },
    }
}

fn gen_rval_expr(fctx: &mut FuncCtx, in_expr: &Expr) -> RVal {
    match gen_expr(fctx, in_expr) {
        Val::LVal(lval) => lval.to_rval(fctx),
        Val::RVal(rval) => rval,
    }
}

fn gen_lval_expr(fctx: &mut FuncCtx, in_expr: &Expr) -> LVal {
    match gen_expr(fctx, in_expr) {
        Val::LVal(lval) => lval,
        _ => panic!("Expected lvalue expression!"),
    }
}

fn memwidth(dtype: &Type) -> &str {
    match dtype {
        Type::U8  => "byte",
        Type::I8  => "byte",
        Type::U16 => "word",
        Type::I16 => "word",
        Type::U32 => "dword",
        Type::I32 => "dword",
        Type::U64 => "qword",
        Type::I64 => "qword",
        Type::Ptr {..} => "qword",
        _ => unreachable!(),
    }
}

fn gen_set(dest: LVal, src: RVal, fctx: &mut FuncCtx) {
    // Write source value to destination
    match dest {
        LVal::Stack(dtype, offset) => {
            fctx.addasm(format!("mov {} [rbp - {}], {}",
                                    memwidth(&dtype), fctx.frame_size - offset,
                                    src.to_string()));
        },
        LVal::Static(dtype, name, offset) => {
            fctx.addasm(format!("mov {} [{} + {}], {}",
                                    memwidth(&dtype), name, offset,
                                    src.to_string()));
        },
        LVal::Deref(dtype, ptr, offset) => {
            fctx.addasm(format!("mov {} [{} + {}], {}",
                                    memwidth(&dtype), ptr.to_string(), offset,
                                    src.to_string()));
        },
    }

    // If the source was a register, make sure it get's deallocated
    Val::RVal(src).discard(fctx);
}

fn gen_init(dest: LVal, init: &Init, fctx: &mut FuncCtx) {

    match dest.get_type() {
        Type::VOID => panic!("Initializer for void!"),
        Type::Array { elem_type, elem_count } => {
            let init_list = init.want_list();
            if init_list.len() != *elem_count {
                panic!("Invalid array initializer!");
            }
            for (i, elem_init) in init_list.iter().enumerate() {
                let elem_dest = dest.array_element(RVal::Immed(Type::U64, i), fctx);
                gen_init(elem_dest, elem_init, fctx);
            }
        },
        Type::Record(record) => {
            let init_list = init.want_list();
            for (i, (name, _)) in record.fields.iter().enumerate() {
                gen_init(dest.record_field(name), &init_list[i], fctx);
            }
        },
        _ => { // Integer/pointer types
            let src = gen_rval_expr(fctx, init.want_expr());
            gen_set(dest, src, fctx);
        },
    }
}

fn gen_ret(retval: RVal, fctx: &mut FuncCtx) {
    // Move return value to rax
    fctx.addasm(format!("mov rax, {}", retval.to_string()));
    // Make sure later expression won't think the return value register is clobbered
    Val::RVal(retval).discard(fctx);
}

fn gen_jcc(label: &Rc<str>, expr1: &Expr, expr2: &Expr, insn: &str, fctx: &mut FuncCtx) {

    let val1 = gen_rval_expr(fctx, expr1);
    let val2 = gen_rval_expr(fctx, expr2);
    fctx.addasm(format!("cmp {}, {}", val1.to_string(), val2.to_string()));
    fctx.addasm(format!("{} .{}", insn, label));

    // We can de-clobber all registers allocated for temporaries (if any)
    Val::RVal(val1).discard(fctx);
    Val::RVal(val2).discard(fctx);
}

fn gen_func<T: std::io::Write>(file: &File, func: &Func, output: &mut T) {
    print_or_die!(output, "{}:", func.name);

    // Create context
    let mut ctx = FuncCtx {
        file: file,
        usedregs: 0,
        locals: HashMap::new(),
        frame_size: 0,
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

    // Move parameters from registers to stack
    for (i, (name, _)) in func.params.iter().enumerate() {
        let (dtype, offset) = ctx.locals.get(name).unwrap();
        // FIXME1: support more than 6 parameters
        // FIXME2: register allocator breaks if we don't first set the
        //         param register to be clobbered (gen_set will clear it)
        ctx.usedregs |= PARAM_REGS[i] as u16;
        gen_set(LVal::Stack(dtype.clone(), *offset),
                RVal::Reg(dtype.clone(), PARAM_REGS[i]),
                &mut ctx);
    }

    for stmt in &func.stmts {
        match stmt {
            Stmt::Label(label) => {
                print_or_die!(output, ".{}:", label);
            },
            Stmt::Auto(ident, dtype, init) => {
                if let Some(init) = init {
                    let (_, offset) = ctx.locals.get(ident).unwrap();
                    gen_init(LVal::Stack(dtype.clone(), *offset), init,
                        &mut ctx);
                }
            },
            Stmt::Set(dest, expr) => {
                let dest = gen_lval_expr(&mut ctx, dest);
                let src = gen_rval_expr(&mut ctx, expr);
                if dest.get_type() != src.get_type() {
                    panic!("Set statement types differ, left: {:?}, right: {:?}",
                            dest.get_type(), src.get_type());
                }
                gen_set(dest, src, &mut ctx);
            }
            Stmt::Eval(ref expr) => {
                gen_expr(&mut ctx, expr).discard(&mut ctx);
            },
            Stmt::Jmp(label) => {
                print_or_die!(output, "jmp .{}", label);
            },
            Stmt::Ret(maybe_expr) => {
                if let Some(expr) = maybe_expr {
                    let retval = gen_rval_expr(&mut ctx, expr);
                    gen_ret(retval, &mut ctx);
                }
                print_or_die!(output, "jmp .$done");
            },
            Stmt::Jeq(label, expr1, expr2) =>
                gen_jcc(label, expr1, expr2, "je", &mut ctx),
            Stmt::Jl(label, expr1, expr2) =>
                gen_jcc(label, expr1, expr2, "jl", &mut ctx),
            Stmt::Jle(label, expr1, expr2) =>
                gen_jcc(label, expr1, expr2, "jle", &mut ctx),
            Stmt::Jg(label, expr1, expr2) =>
                gen_jcc(label, expr1, expr2, "jg", &mut ctx),
            Stmt::Jge(label, expr1, expr2) =>
                gen_jcc(label, expr1, expr2, "jge", &mut ctx),
        }
    }

    // Generate function epilogue
    print_or_die!(output, ".$done:\nleave\nret");
}

pub fn gen_asm<T: std::io::Write>(input: &File, output: &mut T) {
    let mut exports = Vec::new();
    let mut externs = Vec::new();

    let mut bss = HashMap::new();

    // Generate data
    print_or_die!(output, "section .data");
    for (_, cur_static) in &input.statics {
        match cur_static.vis {
            Vis::Export => exports.push(cur_static.name.clone()),
            Vis::Extern => externs.push(cur_static.name.clone()),
            _ => (),
        };

        match &cur_static.init {
            // Generate static initializer in .data
            Some(init) => {
                print_or_die!(output, "{}:", cur_static.name);
                gen_static_init(&cur_static.dtype, init, &input, output);
            },
            // Take note for bss allocation later
            None => {
                bss.insert(cur_static.name.clone(),
                    cur_static.dtype.get_size());
            },
        };
    }

    // Generate bss
    print_or_die!(output, "section .bss");
    for (name, len) in bss {
        print_or_die!(output, "{}: resb {}", name, len);
    }

    // Generate functions
    print_or_die!(output, "section .text");

    for (_, func) in &input.funcs {
        match func.vis {
            Vis::Export => exports.push(func.name.clone()),
            Vis::Extern => externs.push(func.name.clone()),
            _ => (),
        };
        // Generate body for non-extern function
        if func.vis != Vis::Extern {
            gen_func(&input, &func, output);
        }
    }

    // Generate markers
    for exp in exports {
        print_or_die!(output, "global {}", exp);
    }
    for ext in externs {
        print_or_die!(output, "extern {}", ext);
    }
}

pub fn gen_obj<T: std::io::Write>(input: &File, output: &mut T) {
    let tdir = tempdir().unwrap();
    let tasm = tdir.path().join("asm");
    let tobj = tdir.path().join("obj");

    // Generate assembly
    gen_asm(input, &mut std::fs::File::create(&tasm).unwrap());

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
