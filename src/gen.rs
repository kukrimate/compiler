// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use crate::ast::{Expr,ExprKind,Ty,Stmt,Vis};
use std::cell::RefCell;
use std::fmt::Write;
use std::rc::Rc;

//
// Local variables
//

#[derive(Debug)]
pub struct Local {
    vals: RefCell<Option<Val>>
}

impl Local {
    pub fn new() -> Local {
        Local {
            vals: RefCell::new(None)
        }
    }

    pub fn add_val(&self, val: Val) {
        *self.vals.borrow_mut() = Some(val)
    }

    pub fn any_val(&self) -> Val {
        self.vals.borrow().as_ref().unwrap().clone()
    }
}

//
// Operation widths supported by x86
//

#[derive(Clone,Copy)]
enum Width {
    Byte    = 1,
    Word    = 2,
    DWord   = 4,
    QWord   = 8,
}

fn loc_str(width: Width) -> &'static str {
    match width {
        Width::Byte => "byte",
        Width::Word => "word",
        Width::DWord => "dword",
        Width::QWord => "qword",
    }
}

fn data_str(width: Width) -> &'static str {
    match width {
        Width::Byte => "db",
        Width::Word => "dw",
        Width::DWord => "dd",
        Width::QWord => "dq",
    }
}

// Find the largest width operation possible on size bytes
fn max_width(size: usize) -> Width {
    for width in [ Width::QWord, Width::DWord, Width::Word, Width::Byte ] {
        if width as usize <= size {
            return width;
        }
    }
    unreachable!();
}

// Find the register size used for a parameter
fn type_width(ty: &Ty) -> Width {
    match ty {
        Ty::Bool | Ty::U8 | Ty::I8 => Width::Byte,
        Ty::U16 | Ty::I16 => Width::Word,
        Ty::U32 | Ty::I32 => Width::DWord,
        Ty::U64 | Ty::I64 | Ty::USize | Ty::Ptr{..} => Width::QWord,
        _ => unreachable!(),
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
enum Reg {
    Rax = 0,
    Rbx = 1,
    Rcx = 2,
    Rdx = 3,
    Rsi = 4,
    Rdi = 5,
    R8  = 6,
    R9  = 7,
    R10 = 8,
    R11 = 9,
    R12 = 10,
    R13 = 11,
    R14 = 12,
    R15 = 13,
}


//
// Register parameter order
//

const PARAMS: [Reg; 6] = [ Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9 ];

//
// All usable registers (in preferred allocation order)
//

const ALL_REGS: [Reg; 14] = [
    Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15,           // Callee saved (using these is free)
    Reg::Rsi, Reg::Rdi, Reg::R8, Reg::R9, Reg::R10, Reg::R11,   // Never needed
    Reg::Rcx, Reg::Rdx, Reg::Rax, ];                            // Sometimes needed

fn reg_str(width: Width, reg: Reg) -> &'static str {
    match width {
        Width::Byte
            => ["al", "bl", "cl", "dl", "sil", "dil", "r8b", "r9b", "r10b",
                "r11b", "r12b", "r13b", "r14b", "r15b"][reg as usize],
        Width::Word
            => ["ax", "bx", "cx", "dx", "si", "di", "r8w", "r9w", "r10w",
                "r11w", "r12w", "r13w", "r14w", "r15w"][reg as usize],
        Width::DWord
            => ["eax", "ebx", "ecx", "edx", "esi", "edi", "r8d", "r9d", "r10d",
                "r11d", "r12d", "r13d", "r14d", "r15d"][reg as usize],
        Width::QWord
            => ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10",
                "r11", "r12", "r13", "r14", "r15"][reg as usize],
    }
}

//
// Condition codes for comparisons
//

#[derive(Clone,Copy)]
enum Cond {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

fn cond_str(signed: bool, cond: Cond) -> &'static str {
    match cond {
        Cond::Lt => if signed {
            "jl"
        } else {
            "jb"
        },
        Cond::Le => if signed {
            "jle"
        } else {
            "jbe"
        },
        Cond::Gt => if signed {
            "jg"
        } else {
            "ja"
        },
        Cond::Ge => if signed {
            "jge"
        } else {
            "jae"
        },
        Cond::Eq => "je",
        Cond::Ne => "jne",
    }
}

//
// Promise for a runtime value
//

#[derive(Clone,Debug,Hash,PartialEq,Eq)]
pub enum Val {
    Void,                     // Non-existent value
    Imm(usize),               // Immediate constant
    Loc(usize),               // Reference to stack
    Glo(Rc<str>, usize),      // Reference to symbol
    Deref(Box<Val>, usize),   // De-reference of pointer
}

impl Val {
    fn with_offset(self, add: usize) -> Val {
        match self {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Loc(offset) => Val::Loc(offset + add),
            Val::Glo(name, offset) => Val::Glo(name, offset + add),
            Val::Deref(ptr, offset) => Val::Deref(ptr, offset + add),
        }
    }
}

//
// Code generator implementation
//

pub struct Gen {
    // Current function
    frame_size: usize,
    label_no: usize,
    code: String,
    // Linkage table
    linkage: Vec<(Rc<str>, Vis)>,
    // String literal index
    str_no: usize,
    // Sections
    text: String,
    rodata: String,
    data: String,
    bss: String,
}

impl Gen {
    pub fn new() -> Gen {
        Gen {
            // Function
            frame_size: 0,
            label_no: 0,
            code: String::new(),
            // Global
            linkage: Vec::new(),
            str_no: 0,
            text: String::new(),
            rodata: String::new(),
            data: String::new(),
            bss: String::new(),
        }
    }

    pub fn do_link(&mut self, name: Rc<str>, vis: Vis) {
        self.linkage.push((name, vis));
    }

    pub fn do_string(&mut self, chty: Ty, data: &str) -> (Rc<str>, Ty) {
        // Create assembly symbol
        let name: Rc<str> = format!("str${}", self.str_no).into();
        self.str_no += 1;
        // Generate data
        write!(self.rodata, "{} db ", name).unwrap();
        for byte in data.bytes() {
            write!(self.rodata, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(self.rodata, "0").unwrap();
        // Insert symbol
        let ty = Ty::Array {
            elem_type: Box::new(chty),
            elem_count: Some(data.len())
        };
        (name, ty)
    }

    fn gen_static_init(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Const(val) => {
                // Write constant to data section
                // FIXME: align data
                writeln!(self.data, "{} {}", data_str(type_width(&expr.ty)), val).unwrap();
            },
            ExprKind::Compound(exprs) => {
                for expr in exprs.iter() {
                    self.gen_static_init(expr);
                }
            },
            ExprKind::Ref(expr) => {
                if let ExprKind::Global(name) = &expr.kind {
                    writeln!(self.data, "dq {}", name).unwrap();
                } else {
                    panic!("Expected constant expression")
                }
            },
            _ => panic!("Expected constant expression"),
        }
    }

    pub fn do_data(&mut self, name: &Rc<str>, expr: &Expr) {
        // Generate heading
        writeln!(self.data, "{}:", name).unwrap();
        // Generate data
        self.gen_static_init(expr);
    }

    pub fn do_bss(&mut self, name: &Rc<str>, ty: &Ty) {
        // Allocate bss entry
        // FIXME: align .bss entry
        writeln!(self.bss, "{} resb {}", name, ty.get_size()).unwrap();
    }

    fn stack_alloc(&mut self, size: usize) -> usize {
        // FIXME: align allocation
        let offset = self.frame_size;
        self.frame_size += size;
        offset
    }

    fn alloc_ty(&mut self, ty: &Ty) -> Val {
        Val::Loc(self.stack_alloc(ty.get_size()))
    }

    fn alloc_ptr(&mut self) -> Val {
        Val::Loc(self.stack_alloc(Width::QWord as usize))
    }

    fn next_label(&mut self) -> usize {
        let label = self.label_no;
        self.label_no += 1;
        label
    }

    // Load the address of a value into a register
    fn gen_lea(&mut self, reg: Reg, val: &Val) {
        let dreg = reg_str(Width::QWord, reg);
        match val {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Loc(offset) => writeln!(self.code, "lea {}, [rsp + {}]", dreg, offset).unwrap(),
            Val::Glo(name, offset) => writeln!(self.code, "lea {}, [{} + {}]", dreg, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                if *offset > 0 {
                    writeln!(self.code, "lea {}, [{} + {}]", dreg, dreg, offset).unwrap()
                }
            },
        }
    }

    // Load a value with a certain width to a register
    fn gen_load(&mut self, width: Width, reg: Reg, val: &Val) {
        let dreg = reg_str(width, reg);
        match val {
            Val::Void => unreachable!(),
            Val::Imm(val) => writeln!(self.code, "mov {}, {}", dreg, val).unwrap(),
            Val::Loc(offset) => writeln!(self.code, "mov {}, [rsp + {}]", dreg, offset).unwrap(),
            Val::Glo(name, offset) => writeln!(self.code, "mov {}, [{} + {}]", dreg, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                writeln!(self.code, "mov {}, [{} + {}]", dreg, reg_str(Width::QWord, reg), offset).unwrap()
            },
        }
    }

    // Store a register's contents into memory
    fn gen_store(&mut self, width: Width, val: &Val, reg: Reg, tmp_reg: Reg) {
        let sreg = reg_str(width, reg);
        match val {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Loc(offset) => writeln!(self.code, "mov [rsp + {}], {}", offset, sreg).unwrap(),
            Val::Glo(name, offset) => writeln!(self.code, "mov [{} + {}], {}", name, offset, sreg).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, tmp_reg, ptr);
                writeln!(self.code, "mov [{} + {}], {}",
                    reg_str(Width::QWord, tmp_reg), offset, sreg).unwrap()
            },
        }
    }

    // Copy size bytes between two locations
    fn gen_copy(&mut self, mut dst: Val, mut src: Val, mut size: usize) {
        while size > 0 {
            // Find the maximum width we can copy
            let width = max_width(size);

            // Do the copy
            self.gen_load(width, Reg::Rax, &src);
            self.gen_store(width, &dst, Reg::Rax, Reg::Rbx);

            // Adjust for the next step
            size -= width as usize;
            if size > 0 {
                src = src.with_offset(width as usize);
                dst = dst.with_offset(width as usize);
            }
        }
    }

    // Load an arithmetic value
    fn gen_arith_load(&mut self, reg: Reg, ty: &Ty, val: &Val) -> &'static str {
        // Which instruction do we need, and what width do we extend to?
        let (insn, dreg, sloc) = match ty {
            // 8-bit/16-bit types extend to 32-bits
            Ty::U8 => ("movzx", reg_str(Width::DWord, reg), loc_str(Width::Byte)),
            Ty::I8 => ("movsx", reg_str(Width::DWord, reg), loc_str(Width::Byte)),
            Ty::U16 => ("movzx", reg_str(Width::DWord, reg), loc_str(Width::Word)),
            Ty::I16 => ("movsx", reg_str(Width::DWord, reg), loc_str(Width::Word)),
            // 32-bit and 64-bit types don't get extended
            Ty::U32|Ty::I32 => ("mov", reg_str(Width::DWord, reg), loc_str(Width::DWord)),
            Ty::U64|Ty::I64|Ty::USize => ("mov", reg_str(Width::QWord, reg), loc_str(Width::QWord)),
            _ => panic!("Expected arithmetic type"),
        };
        match val {
            Val::Void => unreachable!(),
            Val::Imm(val)
                => writeln!(self.code, "mov {}, {}", dreg, val).unwrap(),
            Val::Loc(offset)
                => writeln!(self.code, "{} {}, {} [rsp + {}]", insn, dreg, sloc, offset).unwrap(),
            Val::Glo(name, offset)
                => writeln!(self.code, "{} {}, {} [{} + {}]", insn, dreg, sloc, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                writeln!(self.code, "{} {}, {} [{} + {}]",
                    insn, dreg, sloc, reg_str(Width::QWord, reg), offset).unwrap()
            },
        }
        dreg
    }

    fn gen_unary(&mut self, op: &str, inner: &Expr) -> Val {
        let val = self.gen_expr(inner);
        let reg = self.gen_arith_load(Reg::Rax, &inner.ty, &val);
        writeln!(self.code, "{} {}", op, reg).unwrap();

        let tmp = self.alloc_ty(&inner.ty);
        self.gen_store(type_width(&inner.ty), &tmp, Reg::Rax, Reg::Rbx);
        tmp
    }

    fn gen_binary(&mut self, op: &str, lhs: &Expr, rhs: &Expr) -> Val {
        // Evaluate operands
        let lhs_val = self.gen_expr(lhs);
        let rhs_val = self.gen_expr(rhs); // NOTE: two types must be equal

        // Do operation
        let lhs_reg = self.gen_arith_load(Reg::Rax, &lhs.ty, &lhs_val);
        let rhs_reg = self.gen_arith_load(Reg::Rbx, &rhs.ty, &rhs_val);
        writeln!(self.code, "{} {}, {}", op, lhs_reg, rhs_reg).unwrap();

        // Save result to temporary
        let tmp = self.alloc_ty(&lhs.ty);
        self.gen_store(type_width(&lhs.ty), &tmp, Reg::Rax, Reg::Rbx);
        tmp
    }

    fn gen_shift(&mut self, op: &str, lhs: &Expr, rhs: &Expr) -> Val {
        // Evaluate operands
        let lhs_val = self.gen_expr(lhs);
        let rhs_val = self.gen_expr(rhs);

        // Do operation
        let lhs_reg = self.gen_arith_load(Reg::Rax, &lhs.ty, &lhs_val);
        self.gen_arith_load(Reg::Rcx, &rhs.ty, &rhs_val);
        writeln!(self.code, "{} {}, cl", op, lhs_reg).unwrap();

        // Save result to temporary
        let tmp = self.alloc_ty(&lhs.ty);
        self.gen_store(type_width(&lhs.ty), &tmp, Reg::Rax, Reg::Rbx);
        tmp
    }

    fn gen_divmod(&mut self, is_mod: bool, lhs: &Expr, rhs: &Expr) -> Val {
        // Evaluate operands
        let lhs_val = self.gen_expr(lhs);
        let rhs_val = self.gen_expr(rhs);

        // Do operation
        self.gen_arith_load(Reg::Rax, &lhs.ty, &lhs_val);
        let rhs_reg = self.gen_arith_load(Reg::Rbx, &rhs.ty, &rhs_val);

        // x86 only has full-division with the upper half in dx
        writeln!(self.code, "xor edx, edx").unwrap();
        // Division also differs based on type
        if lhs.ty.is_signed() {
            writeln!(self.code, "idiv {}", rhs_reg).unwrap();
        } else {
            writeln!(self.code, "div {}", rhs_reg).unwrap();
        }

        // Save result to temporary
        let tmp = self.alloc_ty(&lhs.ty);
        if is_mod { // Remainder in DX
            self.gen_store(type_width(&lhs.ty), &tmp, Reg::Rdx, Reg::Rbx);
        } else {    // Quotient in AX
            self.gen_store(type_width(&lhs.ty), &tmp, Reg::Rax, Reg::Rbx);
        }
        tmp
    }

    fn gen_expr(&mut self, expr: &Expr) -> Val {
        match &expr.kind {
            // Constant value
            ExprKind::Const(val) => Val::Imm(*val),
            // Compound literals
            ExprKind::Compound(exprs) => {
                let off = self.stack_alloc(expr.ty.get_size());
                let mut cur = off;
                for expr in exprs.iter() {
                    let val = self.gen_expr(expr);
                    self.gen_copy(Val::Loc(cur), val, expr.ty.get_size());
                    cur += expr.ty.get_size();
                }
                Val::Loc(off)
            },
            // Reference to symbol
            ExprKind::Global(name)
                => Val::Glo(name.clone(), 0),
            ExprKind::Local(local)
                => local.any_val(),

            // Prefix expressions
            ExprKind::Ref(inner) => {
                // Save address to rax
                let val = self.gen_expr(&*inner);
                self.gen_lea(Reg::Rax, &val);
                // Save pointer to temporary
                let tmp = self.alloc_ty(&expr.ty);
                self.gen_store(type_width(&expr.ty), &tmp, Reg::Rax, Reg::Rbx);
                tmp
            },
            ExprKind::Deref(inner) =>
                Val::Deref(Box::new(self.gen_expr(&*inner)), 0),

            ExprKind::Not(expr)
                => self.gen_unary("not", &*expr),
            ExprKind::Neg(expr)
                => self.gen_unary("neg", &*expr),

            // Postfix expressions
            ExprKind::Field(inner, off)
                => self.gen_expr(&*inner).with_offset(*off),

            ExprKind::Elem(array, index) => {
                // Generate array
                let array_val = self.gen_expr(&*array);

                // Generate index
                let index_val = self.gen_expr(&*index);
                // Save element size (index is multiplied by this)
                let elem_size = expr.ty.get_size();

                if let Val::Imm(index) = index_val {
                    // Avoid emitting multiply on constant index
                    array_val.with_offset(index * elem_size)
                } else {
                    // Generate a de-reference lvalue from a pointer to the element
                    self.gen_lea(Reg::Rax, &array_val);
                    self.gen_load(type_width(&Ty::USize), Reg::Rbx, &index_val);
                    writeln!(self.code, "imul rbx, {}\nadd rax, rbx", elem_size).unwrap();

                    // Allocate temporary
                    let tmp = self.alloc_ptr();
                    self.gen_store(Width::QWord, &tmp, Reg::Rax, Reg::Rbx);
                    Val::Deref(Box::new(tmp), 0)
                }
            },
            ExprKind::Call(func, args) => {
                // Evaluate called expression
                let func_val = self.gen_expr(&*func);

                // Move arguments to registers
                // FIXME: more than 6 arguments
                for (arg, reg) in args.iter().zip(PARAMS) {
                    let arg_val = self.gen_expr(arg);
                    self.gen_load(type_width(&arg.ty), reg, &arg_val);
                }

                // Move function address to rax
                self.gen_lea(Reg::Rbx, &func_val);

                // Verify type is actually a function
                if let Ty::Func { varargs, rettype, .. } = &func.ty {
                    // Generate call
                    if *varargs {
                        writeln!(self.code, "xor eax, eax").unwrap();
                    }
                    writeln!(self.code, "call rbx").unwrap();

                    if let Ty::Void = &**rettype {
                        // Create unusable value for precude
                        Val::Void
                    } else {
                        // Otherwise move returned value to temporary
                        let tmp = self.alloc_ty(&rettype);
                        self.gen_store(type_width(&rettype), &tmp, Reg::Rax, Reg::Rbx);
                        tmp
                    }
                } else {
                    panic!("Non-function object called")
                }
            },

            // Binary operations
            ExprKind::Add(lhs, rhs)
                => self.gen_binary("add", &*lhs, &*rhs),
            ExprKind::Sub(lhs, rhs)
                => self.gen_binary("sub", &*lhs, &*rhs),
            ExprKind::Mul(lhs, rhs)
                => self.gen_binary("imul", &*lhs, &*rhs),
            ExprKind::Div(lhs, rhs)
                => self.gen_divmod(false, &*lhs, &*rhs),
            ExprKind::Rem(lhs, rhs)
                => self.gen_divmod(true, &*lhs, &*rhs),
            ExprKind::Or(lhs, rhs)
                => self.gen_binary("or", &*lhs, &*rhs),
            ExprKind::And(lhs, rhs)
                => self.gen_binary("and", &*lhs, &*rhs),
            ExprKind::Xor(lhs, rhs)
                => self.gen_binary("xor", &*lhs, &*rhs),
            ExprKind::Lsh(lhs, rhs)
                => self.gen_shift("shl", &*lhs, &*rhs),
            ExprKind::Rsh(lhs, rhs)
                => self.gen_shift("shr", &*lhs, &*rhs),

            // Boolean expressions
            ExprKind::LNot(_) |
            ExprKind::Lt(_, _) |
            ExprKind::Le(_, _) |
            ExprKind::Gt(_, _) |
            ExprKind::Ge(_, _) |
            ExprKind::Eq(_, _) |
            ExprKind::Ne(_, _) |
            ExprKind::LAnd(_, _) |
            ExprKind::LOr(_, _) => {
                let ltrue = self.next_label();
                let lfalse = self.next_label();
                let lend = self.next_label();
                self.gen_bool_expr(expr, ltrue, lfalse);

                let off = self.stack_alloc(Ty::Bool.get_size());

                // True case
                writeln!(self.code, ".{}:", ltrue).unwrap();
                writeln!(self.code, "mov byte [rsp + {}], 1", off).unwrap();
                writeln!(self.code, "jmp .{}", lend).unwrap();
                // False case
                writeln!(self.code, ".{}:", lfalse).unwrap();
                writeln!(self.code, "mov byte [rsp + {}], 0", off).unwrap();
                writeln!(self.code, ".{}:", lend).unwrap();

                Val::Loc(off)
            },

            // Cast
            ExprKind::Cast(inner) => {
                // FIXME: integer casts cannot be done this way
                self.gen_expr(&*inner)
            },
        }
    }

    fn gen_jcc(&mut self, cond: Cond, label: usize, lhs: &Expr, rhs: &Expr) {
        let lhs_val = self.gen_expr(lhs);
        let rhs_val = self.gen_expr(rhs);

        let lhs_reg = self.gen_arith_load(Reg::Rax, &lhs.ty, &lhs_val);
        let rhs_reg = self.gen_arith_load(Reg::Rbx, &rhs.ty, &rhs_val);

        writeln!(self.code, "cmp {}, {}\n{} .{}", lhs_reg, rhs_reg,
            cond_str(lhs.ty.is_signed(), cond), label).unwrap();

    }

    fn gen_bool_expr(&mut self, expr: &Expr, ltrue: usize, lfalse: usize) {
        match &expr.kind {
            ExprKind::LNot(inner)
                => self.gen_bool_expr(&*inner, lfalse, ltrue),

            ExprKind::Lt(lhs, rhs) => {
                self.gen_jcc(Cond::Lt, ltrue, &*lhs, &*rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Le(lhs, rhs) => {
                self.gen_jcc(Cond::Le, ltrue, &*lhs, &*rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Gt(lhs, rhs) => {
                self.gen_jcc(Cond::Gt, ltrue, &*lhs, &*rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Ge(lhs, rhs) => {
                self.gen_jcc(Cond::Ge, ltrue, &*lhs, &*rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Eq(lhs, rhs) => {
                self.gen_jcc(Cond::Eq, ltrue, &*lhs, &*rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Ne(lhs, rhs) => {
                self.gen_jcc(Cond::Ne, ltrue, &*lhs, &*rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::LAnd(lhs, rhs) => {
                let lmid = self.next_label();
                self.gen_bool_expr(&*lhs, lmid, lfalse);
                writeln!(self.code, ".{}:", lmid).unwrap();
                self.gen_bool_expr(&*rhs, ltrue, lfalse);
            },
            ExprKind::LOr(lhs, rhs) => {
                let lmid = self.next_label();
                self.gen_bool_expr(&*lhs, ltrue, lmid);
                writeln!(self.code, ".{}:", lmid).unwrap();
                self.gen_bool_expr(&*rhs, ltrue, lfalse);
            }
            _ => {
                // NOTE: we know expr has type bool
                let val = self.gen_expr(expr);
                self.gen_load(Width::Byte, Reg::Rax, &val);
                writeln!(self.code, "test al, al").unwrap();
                writeln!(self.code, "jnz .{}\njmp .{}", ltrue, lfalse).unwrap();
            },
        }
    }

    fn gen_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Block(stmts) => {
                self.gen_stmts(stmts);
            },
            Stmt::Eval(expr) => {
                self.gen_expr(expr);
            },
            Stmt::Ret(opt_expr) => {
                // Evaluate return value if present
                if let Some(expr) = opt_expr {
                    let val = self.gen_expr(expr);
                    self.gen_load(type_width(&expr.ty), Reg::Rax, &val);
                }
                // Then jump to the end of function
                writeln!(self.code, "jmp .$done").unwrap();
            },
            Stmt::Auto(ty, local, opt_expr) => {
                let val = self.alloc_ty(ty);

                // Add value to symbol
                local.add_val(val.clone());

                // Initialze variable if required
                if let Some(expr) = opt_expr {
                    let src = self.gen_expr(expr);
                    self.gen_copy(val, src, expr.ty.get_size());
                }
            },
            Stmt::Label(label)
                => writeln!(self.code, ".{}:", label).unwrap(),
            Stmt::Set(dst, src) => {
                // Find source and destination value
                let dval = self.gen_expr(dst);
                let sval = self.gen_expr(src);
                // Perform copy
                self.gen_copy(dval, sval, dst.ty.get_size());
            },
            Stmt::Jmp(label)
                => writeln!(self.code, "jmp .{}", label).unwrap(),
            Stmt::If(cond, then, opt_else) => {
                let lthen = self.next_label();
                let lelse = self.next_label();
                let lend = self.next_label();

                // Generate conditional
                self.gen_bool_expr(cond, lthen, lelse);

                // Generate true case
                writeln!(self.code, ".{}:", lthen).unwrap();
                self.gen_stmt(then);
                writeln!(self.code, "jmp .{}", lend).unwrap();

                // Generate else case
                writeln!(self.code, ".{}:", lelse).unwrap();
                if let Some(_else) = opt_else {
                    self.gen_stmt(_else);
                }

                writeln!(self.code, ".{}:", lend).unwrap();
            },
            Stmt::While(cond, body) => {
                let ltest = self.next_label();
                let lbody = self.next_label();
                let lend = self.next_label();

                // Generate conditional
                writeln!(self.code, ".{}:", ltest).unwrap();
                self.gen_bool_expr(cond, lbody, lend);

                // Generate body
                writeln!(self.code, ".{}:", lbody).unwrap();
                self.gen_stmt(body);

                writeln!(self.code, "jmp .{}\n.{}:", ltest, lend).unwrap();
            },
        }
    }

    fn gen_stmts(&mut self, stmts: &Vec<Stmt>) {
        for stmt in stmts {
            self.gen_stmt(stmt);
        }
    }

    pub fn do_func(&mut self, name: Rc<str>, params: Vec<(Ty, Rc<Local>)>, stmts: Vec<Stmt>) {
        // Reset function specific fields
        self.frame_size = 0;
        self.label_no = 0;
        self.code.clear();

        // Generate heading
        writeln!(self.text, "{}:", name).unwrap();

        // Copy the parameters into locals
        // FIXME: this doesn't take into account most of the SysV ABI
        for (i, (ty, local)) in params.into_iter().enumerate() {
            let val = self.alloc_ty(&ty);
            // Copy parameter to local variable
            self.gen_store(type_width(&ty), &val, PARAMS[i], Reg::Rbx);
            // Add local variable to parameter symbol
            local.add_val(val);
        }

        // Generate statements
        self.gen_stmts(&stmts);

        // Round stack frame
        self.frame_size = (self.frame_size + 15) / 16 * 16;
        // Generate code
        writeln!(self.text, "push rbp\nmov rbp, rsp\nsub rsp, {}\n{}.$done:\nleave\nret",
            self.frame_size, self.code).unwrap();
    }

    pub fn finalize<T: std::io::Write>(&self, output: &mut T) {
        // Write sections
        writeln!(output, "section .text\n{}", self.text).unwrap();
        writeln!(output, "section .rodata\n{}", self.rodata).unwrap();
        writeln!(output, "section .data\n{}", self.data).unwrap();
        writeln!(output, "section .bss\n{}", self.bss).unwrap();

        // Generate import/export table
        for (name, vis) in &self.linkage {
            match vis {
                Vis::Private => (),
                Vis::Export => writeln!(output, "global {}", name).unwrap(),
                Vis::Extern => writeln!(output, "extern {}", name).unwrap(),
            }
        }
    }
}
