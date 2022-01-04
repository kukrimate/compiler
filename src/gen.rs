// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use crate::ast::{Expr,ExprKind,Ty,Stmt,Vis};
use std::collections::HashMap;
use std::fmt::Write;
use std::rc::Rc;

fn is_signed(ty: &Ty) -> bool {
    match ty {
        Ty::I8|Ty::I16|Ty::I32|Ty::I64 => true,
        _ => false
    }
}

#[derive(Clone,Copy)]
enum Width {
    Byte    = 1,
    Word    = 2,
    DWord   = 4,
    QWord   = 8,
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

#[derive(Clone,Copy)]
enum Reg {
    Rax = 0,
    Rbx = 1,
    Rcx = 2,
    Rdx = 3,
    Rsi = 4,
    Rdi = 5,
    R8  = 6,
    R9  = 7,
    /*R10 = 8,
    R11 = 9,
    R12 = 10,
    R13 = 11,
    R14 = 12,
    R15 = 13,*/
}

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

fn loc_str(width: Width) -> &'static str {
    match width {
        Width::Byte => "byte",
        Width::Word => "word",
        Width::DWord => "dword",
        Width::QWord => "qword",
    }
}

impl Reg {
    fn to_str(&self, dtype: &Ty) -> &str {
        match dtype {
            Ty::Bool|Ty::U8|Ty::I8
                => ["al", "bl", "cl", "dl", "sil", "dil", "r8b", "r9b", "r10b",
                    "r11b", "r12b", "r13b", "r14b", "r15b"][*self as usize],
            Ty::U16|Ty::I16
                => ["ax", "bx", "cx", "dx", "si", "di", "r8w", "r9w", "r10w",
                    "r11w", "r12w", "r13w", "r14w", "r15w"][*self as usize],
            Ty::U32|Ty::I32
                => ["eax", "ebx", "ecx", "edx", "esi", "edi", "r8d", "r9d", "r10d",
                    "r11d", "r12d", "r13d", "r14d", "r15d"][*self as usize],
            Ty::U64|Ty::I64|Ty::USize|Ty::Ptr{..}
                => ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10",
                    "r11", "r12", "r13", "r14", "r15"][*self as usize],
            _
                => panic!("Read or write to non-primtive type {:?}", dtype),
        }
    }
}

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

fn asm_dataword(dtype: &Ty) -> &str {
    match dtype {
        Ty::U8|Ty::I8 => "db",
        Ty::U16|Ty::I16 => "dw",
        Ty::U32|Ty::I32 => "dd",
        Ty::U64|Ty::I64|Ty::Ptr{..} => "dq",
        _ => unreachable!(),
    }
}

const PARAMS: [Reg; 6] = [ Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9 ];

enum SymKind {
    Global(Vis),
    Local(usize),
}

struct Sym {
    dtype: Ty,
    kind: SymKind,
}

impl Sym {
    fn make_global(dtype: Ty, vis: Vis) -> Sym {
        Sym {
            dtype: dtype,
            kind: SymKind::Global(vis),
        }
    }

    fn make_local(dtype: Ty, offset: usize) -> Sym {
        Sym {
            dtype: dtype,
            kind: SymKind::Local(offset),
        }
    }
}

//
// Chained hash tables used for a symbol table
//

struct SymTab {
    list: Vec<HashMap<Rc<str>, Sym>>,
}

impl SymTab {
    fn new() -> SymTab {
        let mut cm = SymTab {
            list: Vec::new(),
        };
        cm.list.push(HashMap::new());
        cm
    }

    fn insert(&mut self, name: Rc<str>, sym: Sym) {
        if let Some(inner_scope) = self.list.last_mut() {
            if let Some(_) = inner_scope.insert(name.clone(), sym) {
                panic!("Re-declaration of {}", name)
            }
        } else {
            unreachable!();
        }
    }

    fn lookup(&mut self, name: &Rc<str>) -> &Sym {
        for scope in self.list.iter().rev() {
            if let Some(sym) = scope.get(name) {
                return sym;
            }
        }
        panic!("Unknown identifier {}", name)
    }

    fn push_scope(&mut self) {
        self.list.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        if self.list.len() < 2 {
            unreachable!();
        }
        self.list.pop();
    }
}

//
// Promise for a runtime value
//

#[derive(Clone)]
enum Val {
    Imm(usize),               // Immediate constant
    Off(usize),               // Reference to stack
    Sym(Rc<str>, usize),      // Reference to symbol
    Deref(Box<Val>, usize),   // De-reference of pointer
    Void,                     // Non-existent value
}

impl Val {
    fn ptr_to_reg(&self, text: &mut String, reg: Reg) {
        let void_ptr = Ty::Ptr { base_type: Box::new(Ty::Void) };
        match self {
            Val::Void => panic!("Use of void value"),
            Val::Imm(_) => panic!("Cannot take address of immediate"),
            Val::Off(offset) => writeln!(text, "lea {}, [rsp + {}]", reg.to_str(&void_ptr), offset).unwrap(),
            Val::Sym(name, offset) => writeln!(text, "lea {}, [{} + {}]", reg.to_str(&void_ptr), name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                ptr.val_to_reg(text, &void_ptr, reg);
                if *offset > 0 {
                    writeln!(text, "lea {}, [{} + {}]",
                        reg.to_str(&void_ptr), reg.to_str(&void_ptr), offset).unwrap();
                }
            },
        }
    }

    fn val_to_reg(&self, text: &mut String, dtype: &Ty, reg: Reg) {
        let void_ptr = Ty::Ptr { base_type: Box::new(Ty::Void) };
        match self {
            Val::Void => panic!("Use of void value"),
            Val::Imm(val)
                => writeln!(text, "mov {}, {}", reg.to_str(dtype), val).unwrap(),
            Val::Off(offset)
                => writeln!(text, "mov {}, [rsp + {}]", reg.to_str(dtype), offset).unwrap(),
            Val::Sym(name, offset)
                => writeln!(text, "mov {}, [{} + {}]", reg.to_str(dtype), name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                ptr.val_to_reg(text, &void_ptr, reg);
                writeln!(text, "mov {}, [{} + {}]",
                    reg.to_str(dtype), reg.to_str(&void_ptr), offset).unwrap();
            },
        }
    }

    fn with_offset(&self, add: usize) -> Val {
        match self {
            Val::Void => panic!("Use of void value"),
            Val::Imm(_) => panic!("Offset from immediate"),
            Val::Off(offset) => Val::Off(offset + add),
            Val::Sym(name, offset) => Val::Sym(name.clone(), offset + add),
            Val::Deref(ptr, offset) => Val::Deref(ptr.clone(), offset + add),
        }
    }
}

pub struct Gen {
    // Current function
    frame_size: usize,
    label_no: usize,
    code: String,
    // Symbol table
    symtab: SymTab,
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
            symtab: SymTab::new(),
            str_no: 0,
            text: String::new(),
            rodata: String::new(),
            data: String::new(),
            bss: String::new(),
        }
    }

    pub fn do_sym(&mut self, vis: Vis, name: Rc<str>, dtype: Ty) {
        self.symtab.insert(name.clone(), Sym::make_global(dtype, vis))
    }

    pub fn do_string(&mut self, chty: Ty, data: &str) -> (Rc<str>, Ty) {
        // Create assembly symbol
        let name: Rc<str> = format!("str${}", self.str_no).into();
        self.str_no += 1;
        // Generate data
        write!(self.rodata, "{} {} ", name, asm_dataword(&chty)).unwrap();
        for byte in data.bytes() {
            write!(self.rodata, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(self.rodata, "0").unwrap();
        // Insert symbol
        let ty = Ty::Array {
            elem_type: Box::new(chty),
            elem_count: Some(data.len())
        };
        self.do_sym(Vis::Private, name.clone(), ty.clone());
        (name, ty)
    }

    fn gen_static_init(&mut self, expr: Expr) {
        match expr.kind {
            ExprKind::Const(val) => {
                // Write constant to data section
                // FIXME: align data
                writeln!(self.data, "{} {}", asm_dataword(&expr.ty), val).unwrap();
            },
            ExprKind::Compound(exprs) => {
                for expr in exprs.into_iter() {
                    self.gen_static_init(expr);
                }
            },
            ExprKind::Ref(expr) => {
                if let ExprKind::Sym(name) = expr.kind {
                    writeln!(self.data, "dq {}", name).unwrap();
                } else {
                    panic!("Expected constant expression")
                }
            },
            _ => panic!("Expected constant expression"),
        }
    }

    pub fn do_static_init(&mut self, vis: Vis, name: Rc<str>, ty: Ty, expr: Expr) {
        // Generate heading
        writeln!(self.data, "{}:", name).unwrap();
        // Generate data
        self.gen_static_init(expr);
        // Create symbol
        self.do_sym(vis, name, ty);
    }

    pub fn do_static(&mut self, vis: Vis, name: Rc<str>, dtype: Ty) {
        // Allocate bss entry
        // FIXME: align .bss entry
        writeln!(self.bss, "{} resb {}", name, dtype.get_size()).unwrap();
        // Create symbol
        self.do_sym(vis, name, dtype);
    }

    fn stack_alloc(&mut self, dtype: &Ty) -> usize {
        // FIXME: align allocation
        let offset = self.frame_size;
        self.frame_size += dtype.get_size();
        offset
    }

    fn alloc_temporary(&mut self, dtype: &Ty) -> Val {
        let offset = self.stack_alloc(dtype);
        Val::Off(offset)
    }

    fn next_label(&mut self) -> usize {
        let label = self.label_no;
        self.label_no += 1;
        label
    }

    // Load a value with a certain width to a register
    fn gen_load(&mut self, width: Width, reg: Reg, val: &Val) {
        let dreg = reg_str(width, reg);
        match val {
            Val::Void => unreachable!(),
            Val::Imm(val) => writeln!(self.code, "mov {}, {}", dreg, val).unwrap(),
            Val::Off(offset) => writeln!(self.code, "mov {}, [rsp + {}]", dreg, offset).unwrap(),
            Val::Sym(name, offset) => writeln!(self.code, "mov {}, [{} + {}]", dreg, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                writeln!(self.code, "mov {}, [{} + {}]", dreg, dreg, offset).unwrap()
            },
        }
    }

    // Store a register's contents into memory
    fn gen_store(&mut self, width: Width, val: &Val, reg: Reg, tmp_reg: Reg) {
        let sreg = reg_str(width, reg);
        match val {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Off(offset) => writeln!(self.code, "mov [rsp + {}], {}", offset, sreg).unwrap(),
            Val::Sym(name, offset) => writeln!(self.code, "mov [{} + {}], {}", name, offset, sreg).unwrap(),
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
            Val::Off(offset)
                => writeln!(self.code, "{} {}, {} [rsp + {}]", insn, dreg, sloc, offset).unwrap(),
            Val::Sym(name, offset)
                => writeln!(self.code, "{} {}, {} [{} + {}]", insn, dreg, sloc, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                writeln!(self.code, "{} {}, {} [{} + {}]",
                    insn, dreg, sloc, reg_str(Width::QWord, reg), offset).unwrap()
            },
        }
        dreg
    }

    fn gen_unary(&mut self, op: &str, expr: Expr) -> (Ty, Val) {
        let (ty, val) = self.gen_expr(expr);
        let reg = self.gen_arith_load(Reg::Rax, &ty, &val);
        writeln!(self.code, "{} {}", op, reg).unwrap();

        let tmp = self.alloc_temporary(&ty);
        self.gen_store(type_width(&ty), &tmp, Reg::Rax, Reg::Rbx);
        (ty, tmp)
    }

    fn gen_binary(&mut self, op: &str, lhs: Expr, rhs: Expr) -> (Ty, Val) {
        // Evaluate operands
        let (lhs_ty, lhs_val) = self.gen_expr(lhs);
        let (rhs_ty, rhs_val) = self.gen_expr(rhs); // NOTE: two types must be equal

        // Do operation
        let lhs_reg = self.gen_arith_load(Reg::Rax, &lhs_ty, &lhs_val);
        let rhs_reg = self.gen_arith_load(Reg::Rbx, &rhs_ty, &rhs_val);
        writeln!(self.code, "{} {}, {}", op, lhs_reg, rhs_reg).unwrap();

        // Save result to temporary
        let tmp = self.alloc_temporary(&lhs_ty);
        self.gen_store(type_width(&lhs_ty), &tmp, Reg::Rax, Reg::Rbx);
        (lhs_ty, tmp)
    }

    fn gen_shift(&mut self, op: &str, lhs: Expr, rhs: Expr) -> (Ty, Val) {
        // Evaluate operands
        let (lhs_ty, lhs_val) = self.gen_expr(lhs);
        let (rhs_ty, rhs_val) = self.gen_expr(rhs);

        // Do operation
        let lhs_reg = self.gen_arith_load(Reg::Rax, &lhs_ty, &lhs_val);
        self.gen_arith_load(Reg::Rcx, &rhs_ty, &rhs_val);
        writeln!(self.code, "{} {}, cl", op, lhs_reg).unwrap();

        // Save result to temporary
        let tmp = self.alloc_temporary(&lhs_ty);
        self.gen_store(type_width(&lhs_ty), &tmp, Reg::Rax, Reg::Rbx);
        (lhs_ty, tmp)
    }

    fn gen_divmod(&mut self, is_mod: bool, lhs: Expr, rhs: Expr) -> (Ty, Val) {
        // Evaluate operands
        let (lhs_ty, lhs_val) = self.gen_expr(lhs);
        let (rhs_ty, rhs_val) = self.gen_expr(rhs);

        // Do operation
        self.gen_arith_load(Reg::Rax, &lhs_ty, &lhs_val);
        let rhs_reg = self.gen_arith_load(Reg::Rbx, &rhs_ty, &rhs_val);

        // x86 only has full-division with the upper half in dx
        writeln!(self.code, "xor edx, edx").unwrap();
        // Division also differs based on type
        if is_signed(&lhs_ty) {
            writeln!(self.code, "idiv {}", rhs_reg).unwrap();
        } else {
            writeln!(self.code, "div {}", rhs_reg).unwrap();
        }

        // Save result to temporary
        let tmp = self.alloc_temporary(&lhs_ty);
        if is_mod { // Remainder in DX
            self.gen_store(type_width(&lhs_ty), &tmp, Reg::Rdx, Reg::Rbx);
        } else {    // Quotient in AX
            self.gen_store(type_width(&lhs_ty), &tmp, Reg::Rax, Reg::Rbx);
        }
        (lhs_ty, tmp)
    }

    fn gen_expr(&mut self, expr: Expr) -> (Ty, Val) {
        match expr.kind {
            // Constant value
            ExprKind::Const(val) => (expr.ty.clone(), Val::Imm(val)),
            // Compound literals
            ExprKind::Compound(exprs) => {
                let off = self.stack_alloc(&expr.ty);
                let mut cur = off;

                for expr in exprs.into_iter() {
                    let (ty, val) = self.gen_expr(expr);
                    self.gen_copy(Val::Off(cur), val, ty.get_size());
                    cur += ty.get_size();
                }

                (expr.ty.clone(), Val::Off(off))
            },
            // Reference to symbol
            ExprKind::Sym(name) => {
                let sym = self.symtab.lookup(&name);
                match sym.kind {
                    SymKind::Global(_)
                        => (sym.dtype.clone(), Val::Sym(name, 0)),
                    SymKind::Local(offset)
                        => (sym.dtype.clone(), Val::Off(offset)),
                }
            },

            // Pointer ref/deref
            ExprKind::Ref(base) => {
                // Save address to rax
                let (base_type, base_val) = self.gen_expr(*base);
                base_val.ptr_to_reg(&mut self.code, Reg::Rax);
                // Create pointer type
                let ty = Ty::Ptr { base_type: Box::new(base_type) };
                // Save pointer to temporary
                let tmp = self.alloc_temporary(&ty);
                self.gen_store(type_width(&ty), &tmp, Reg::Rax, Reg::Rbx);
                (ty, tmp)
            },
            ExprKind::Deref(ptr) => {
                let (ptr_type, ptr_val) = self.gen_expr(*ptr);
                if let Ty::Ptr { base_type } = ptr_type {
                    (*base_type, Val::Deref(Box::new(ptr_val), 0))
                } else {
                    panic!("De-referenced non-pointer type")
                }
            },

            // Unary operations
            ExprKind::Not(expr)
                => self.gen_unary("not", *expr),
            ExprKind::Neg(expr)
                => self.gen_unary("neg", *expr),

            // Postfix expressions
            ExprKind::Field(inner, off) => {
                let (_, val) = self.gen_expr(*inner);
                (expr.ty, val.with_offset(off))
            },
            ExprKind::Elem(array, index) => {
                // Generate array
                let (array_type, array_val) = self.gen_expr(*array);
                let elem_type = if let Ty::Array { elem_type, .. } = array_type {
                    *elem_type
                } else {
                    panic!("Indexed non-array type")
                };
                // Generate index
                let (index_type, index_val) = self.gen_expr(*index);

                if let Val::Imm(val) = index_val {
                    // Constant index is cheaper
                    let offset = val * elem_type.get_size();
                    (elem_type, array_val.with_offset(offset))
                } else {
                    // Generate a de-reference lvalue from a pointer to the element
                    array_val.ptr_to_reg(&mut self.code, Reg::Rax);
                    index_val.val_to_reg(&mut self.code, &index_type, Reg::Rbx);
                    writeln!(&mut self.code, "imul rbx, {}\nadd rax, rbx",
                        elem_type.get_size()).unwrap();

                    // Allocate temporary
                    let ptr_type = Ty::Ptr { base_type: Box::new(elem_type.clone()) };
                    let ptr_val = self.alloc_temporary(&ptr_type);
                    self.gen_store(type_width(&ptr_type), &ptr_val, Reg::Rax, Reg::Rbx);
                    (elem_type, Val::Deref(Box::new(ptr_val), 0))
                }
            },
            ExprKind::Call(func, args) => {
                // Evaluate called expression
                let (func_type, func_val) = self.gen_expr(*func);

                // Move arguments to registers
                // FIXME: more than 6 arguments
                for (arg, reg) in args.into_iter().zip(PARAMS) {
                    let (arg_type, arg_val) = self.gen_expr(arg);
                    arg_val.val_to_reg(&mut self.code, &arg_type, reg);
                }

                // Move function address to rax
                func_val.ptr_to_reg(&mut self.code, Reg::Rbx);
                // Verify type is actually a function
                let (varargs, rettype) = if let Ty::Func
                        { varargs, rettype, .. } = func_type {
                    (varargs, *rettype)
                } else {
                    panic!("Non-function object called")
                };

                // Generate call
                // FIXME: direct call to label should not be indirect
                if varargs {
                    writeln!(&mut self.code, "xor eax, eax").unwrap();
                }
                writeln!(&mut self.code, "call rbx").unwrap();

                if let Ty::Void = rettype {
                    // Create unusable value
                    (rettype, Val::Void)
                } else {
                    // Move return value to temporary
                    let tmp = self.alloc_temporary(&rettype);
                    self.gen_store(type_width(&rettype), &tmp, Reg::Rax, Reg::Rbx);
                    (rettype, tmp)
                }
            },

            // Binary operations
            ExprKind::Add(lhs, rhs)
                => self.gen_binary("add", *lhs, *rhs),
            ExprKind::Sub(lhs, rhs)
                => self.gen_binary("sub", *lhs, *rhs),
            ExprKind::Mul(lhs, rhs)
                => self.gen_binary("imul", *lhs, *rhs),
            ExprKind::Div(lhs, rhs)
                => self.gen_divmod(false, *lhs, *rhs),
            ExprKind::Rem(lhs, rhs)
                => self.gen_divmod(true, *lhs, *rhs),
            ExprKind::Or(lhs, rhs)
                => self.gen_binary("or", *lhs, *rhs),
            ExprKind::And(lhs, rhs)
                => self.gen_binary("and", *lhs, *rhs),
            ExprKind::Xor(lhs, rhs)
                => self.gen_binary("xor", *lhs, *rhs),
            ExprKind::Lsh(lhs, rhs)
                => self.gen_shift("shl", *lhs, *rhs),
            ExprKind::Rsh(lhs, rhs)
                => self.gen_shift("shr", *lhs, *rhs),

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

                let ty = Ty::Bool;
                let off = self.stack_alloc(&ty);

                // True case
                writeln!(self.code, ".{}:", ltrue).unwrap();
                writeln!(self.code, "mov byte [rsp + {}], 1", off).unwrap();
                writeln!(self.code, "jmp .{}", lend).unwrap();
                // False case
                writeln!(self.code, ".{}:", lfalse).unwrap();
                writeln!(self.code, "mov byte [rsp + {}], 0", off).unwrap();
                writeln!(self.code, ".{}:", lend).unwrap();

                (ty, Val::Off(off))
            },

            // Cast
            ExprKind::Cast(inner) => {
                // FIXME: integer casts cannot be done this way
                let (_, val) = self.gen_expr(*inner);
                (expr.ty, val)
            },
        }
    }

    fn gen_jcc(&mut self, cond: Cond, label: usize, lhs: Expr, rhs: Expr) {
        let (lhs_ty, lhs_val) = self.gen_expr(lhs);
        let (rhs_ty, rhs_val) = self.gen_expr(rhs);

        let lhs_reg = self.gen_arith_load(Reg::Rax, &lhs_ty, &lhs_val);
        let rhs_reg = self.gen_arith_load(Reg::Rbx, &rhs_ty, &rhs_val);

        writeln!(self.code, "cmp {}, {}\n{} .{}", lhs_reg, rhs_reg,
            cond_str(is_signed(&lhs_ty), cond), label).unwrap();

    }

    fn gen_bool_expr(&mut self, expr: Expr, ltrue: usize, lfalse: usize) {
        match expr.kind {
            ExprKind::LNot(expr) => self.gen_bool_expr(*expr, lfalse, ltrue),
            ExprKind::Lt(lhs, rhs) => {
                self.gen_jcc(Cond::Lt, ltrue, *lhs, *rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Le(lhs, rhs) => {
                self.gen_jcc(Cond::Le, ltrue, *lhs, *rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Gt(lhs, rhs) => {
                self.gen_jcc(Cond::Gt, ltrue, *lhs, *rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Ge(lhs, rhs) => {
                self.gen_jcc(Cond::Ge, ltrue, *lhs, *rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Eq(lhs, rhs) => {
                self.gen_jcc(Cond::Eq, ltrue, *lhs, *rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::Ne(lhs, rhs) => {
                self.gen_jcc(Cond::Ne, ltrue, *lhs, *rhs);
                writeln!(self.code, "jmp .{}", lfalse).unwrap();
            },
            ExprKind::LAnd(lhs, rhs) => {
                let lmid = self.next_label();
                self.gen_bool_expr(*lhs, lmid, lfalse);
                writeln!(self.code, ".{}:", lmid).unwrap();
                self.gen_bool_expr(*rhs, ltrue, lfalse);
            },
            ExprKind::LOr(lhs, rhs) => {
                let lmid = self.next_label();
                self.gen_bool_expr(*lhs, ltrue, lmid);
                writeln!(self.code, ".{}:", lmid).unwrap();
                self.gen_bool_expr(*rhs, ltrue, lfalse);
            }
            _ => {
                let (ty, val) = self.gen_expr(expr);
                val.val_to_reg(&mut self.code, &ty, Reg::Rax);
                writeln!(self.code, "test {}, {}",
                    Reg::Rax.to_str(&ty),
                    Reg::Rax.to_str(&ty)).unwrap();
                writeln!(self.code, "jnz .{}\njmp .{}", ltrue, lfalse)
                    .unwrap();
            },
        }
    }

    fn gen_stmt(&mut self, stmt: Stmt) {
        match stmt {
            Stmt::Block(stmts) => {
                self.symtab.push_scope();
                self.gen_stmts(stmts);
                self.symtab.pop_scope();
            },
            Stmt::Eval(expr) => {
                self.gen_expr(expr);
            },
            Stmt::Ret(opt_expr) => {
                // Evaluate return value if present
                if let Some(expr) = opt_expr {
                    let (ty, val) = self.gen_expr(expr);
                    val.val_to_reg(&mut self.code, &ty, Reg::Rax);
                }
                // Then jump to the end of function
                writeln!(&mut self.code, "jmp .$done").unwrap();
            },
            Stmt::Auto(name, ty, opt_init) => {
                let off = self.stack_alloc(&ty);
                self.symtab.insert(name, Sym::make_local(ty, off));

                if let Some(init) = opt_init {
                    // Generate initializer
                    let (ty, src) = self.gen_expr(init);
                    // Copy initializer to local
                    self.gen_copy(Val::Off(off), src, ty.get_size());
                }
            },
            Stmt::Label(label)
                => writeln!(&mut self.code, ".{}:", label).unwrap(),
            Stmt::Set(dst, src) => {
                // Find source and destination value
                let (ty, dval) = self.gen_expr(dst);
                let (_, sval) = self.gen_expr(src);
                // Perform copy
                self.gen_copy(dval, sval, ty.get_size());
            },
            Stmt::Jmp(label)
                => writeln!(&mut self.code, "jmp .{}", label).unwrap(),
            Stmt::If(cond, then, opt_else) => {
                let lthen = self.next_label();
                let lelse = self.next_label();
                let lend = self.next_label();

                // Generate conditional
                self.gen_bool_expr(cond, lthen, lelse);

                // Generate true case
                writeln!(self.code, ".{}:", lthen).unwrap();
                self.gen_stmt(*then);
                writeln!(self.code, "jmp .{}", lend).unwrap();

                // Generate else case
                writeln!(self.code, ".{}:", lelse).unwrap();
                if let Some(_else) = opt_else {
                    self.gen_stmt(*_else);
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
                self.gen_stmt(*body);

                writeln!(self.code, "jmp .{}\n.{}:", ltest, lend).unwrap();
            },
        }
    }

    fn gen_stmts(&mut self, stmts: Vec<Stmt>) {
        for stmt in stmts.into_iter() {
            self.gen_stmt(stmt);
        }
    }

    pub fn do_func(&mut self, name: Rc<str>, _rettype: Ty, param_tab: Vec<(Rc<str>, Ty)>, stmts: Vec<Stmt>) {
        self.frame_size = 0;
        self.label_no = 0;
        self.code.clear();

        // Generate heading
        writeln!(self.text, "{}:", name).unwrap();

        // Create function scope
        self.symtab.push_scope();

        // Copy the parameters into locals
        // FIXME: this doesn't take into account most of the SysV ABI
        for (i, (name, ty)) in param_tab.into_iter().enumerate() {
            let off = self.stack_alloc(&ty);
            self.gen_store(type_width(&ty), &Val::Off(off), PARAMS[i], Reg::Rbx);
            self.symtab.insert(name, Sym::make_local(ty, off));
        }

        // Generate statements
        self.gen_stmts(stmts);

        // Round stack frame
        self.frame_size = (self.frame_size + 15) / 16 * 16;
        // Generate code
        writeln!(self.text, "push rbp\nmov rbp, rsp\nsub rsp, {}\n{}.$done:\nleave\nret",
            self.frame_size, self.code).unwrap();
        // Drop function scope
        self.symtab.pop_scope();
    }

    pub fn finalize<T: std::io::Write>(&self, output: &mut T) {
        // Write sections
        writeln!(output, "section .text\n{}", self.text).unwrap();
        writeln!(output, "section .rodata\n{}", self.rodata).unwrap();
        writeln!(output, "section .data\n{}", self.data).unwrap();
        writeln!(output, "section .bss\n{}", self.bss).unwrap();

        // Generate import/export table
        for (name, sym) in self.symtab.list[0].iter() {
            if let SymKind::Global(vis) = &sym.kind {
                match vis {
                    Vis::Private => (),
                    Vis::Export => writeln!(output, "global {}", name).unwrap(),
                    Vis::Extern => writeln!(output, "extern {}", name).unwrap(),
                }
            }
        }
    }
}
