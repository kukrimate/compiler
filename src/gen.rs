// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use crate::ast::{Expr,Init,Type,Stmt,Vis};
use std::collections::HashMap;
use std::fmt::Write;
use std::rc::Rc;

fn asm_dataword(dtype: &Type) -> &str {
    match dtype {
        Type::U8|Type::I8 => "db",
        Type::U16|Type::I16 => "dw",
        Type::U32|Type::I32 => "dd",
        Type::U64|Type::I64 => "dq",
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

impl Reg {
    fn to_str(&self, dtype: &Type) -> &str {
        match dtype {
            Type::U8|Type::I8
                => ["al", "bl", "cl", "dl", "sil", "dil", "r8b", "r9b", "r10b",
                    "r11b", "r12b", "r13b", "r14b", "r15b"][*self as usize],
            Type::U16|Type::I16
                => ["ax", "bx", "cx", "dx", "si", "di", "r8w", "r9w", "r10w",
                    "r11w", "r12w", "r13w", "r14w", "r15w"][*self as usize],
            Type::U32|Type::I32
                => ["eax", "ebx", "ecx", "edx", "esi", "edi", "r8d", "r9d", "r10d",
                    "r11d", "r12d", "r13d", "r14d", "r15d"][*self as usize],
            Type::U64|Type::I64|Type::Ptr{..}
                => ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10",
                    "r11", "r12", "r13", "r14", "r15"][*self as usize],
            Type::Deduce
                => panic!("Type deduction failure, be more specific"),
            _
                => panic!("Read or write to non-primtive type {:?}", dtype),
        }
    }
}

const PARAMS: [Reg; 6] = [ Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9 ];

enum SymKind {
    Global(Vis),
    Local(usize),
}

struct Sym {
    dtype: Type,
    kind: SymKind,
}

impl Sym {
    fn make_global(dtype: Type, vis: Vis) -> Sym {
        Sym {
            dtype: dtype,
            kind: SymKind::Global(vis),
        }
    }

    fn make_local(dtype: Type, offset: usize) -> Sym {
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
        let void_ptr = Type::Ptr { base_type: Box::new(Type::Void) };
        match self {
            Val::Imm(_) => panic!("Cannot take address of immediate"),
            Val::Off(offset) => {
                writeln!(text, "lea {}, [rsp + {}]",
                    reg.to_str(&void_ptr), offset).unwrap();
            },
            Val::Sym(name, offset) => {
                writeln!(text, "lea {}, [{} + {}]",
                    reg.to_str(&void_ptr), name, offset).unwrap();
            },
            Val::Deref(ptr, offset) => {
                ptr.val_to_reg(text, &void_ptr, reg);
                if *offset > 0 {
                    writeln!(text, "lea {}, [{} + {}]",
                        reg.to_str(&void_ptr), reg.to_str(&void_ptr), offset)
                    .unwrap();
                }
            },
            Val::Void => panic!("Use of void value"),
        }
    }

    fn val_to_reg(&self, text: &mut String, dtype: &Type, reg: Reg) {
        let void_ptr = Type::Ptr { base_type: Box::new(Type::Void) };
        match self {
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
            Val::Void => panic!("Use of void value"),
        }
    }

    fn set_from_reg(&self, text: &mut String, dtype: &Type, reg: Reg, tmp_reg: Reg) {
        let void_ptr = Type::Ptr { base_type: Box::new(Type::Void) };
        match self {
            Val::Imm(val)
                => writeln!(text, "mov {}, {}", val, reg.to_str(dtype)).unwrap(),
            Val::Off(offset)
                => writeln!(text, "mov [rsp + {}], {}", offset, reg.to_str(dtype)).unwrap(),
            Val::Sym(name, offset)
                => writeln!(text, "mov [{} + {}], {}", name, offset, reg.to_str(dtype)).unwrap(),
            Val::Deref(ptr, offset) => {
                ptr.val_to_reg(text, &void_ptr, tmp_reg);
                writeln!(text, "mov [{} + {}], {}",
                    tmp_reg.to_str(&void_ptr), offset, reg.to_str(dtype)).unwrap();
            },
            Val::Void => panic!("Use of void value"),
        }
    }

    fn with_offset(&self, add: usize) -> Val {
        match self {
            Val::Off(offset) => Val::Off(offset + add),
            Val::Sym(name, offset) => Val::Sym(name.clone(), offset + add),
            Val::Deref(ptr, offset) => Val::Deref(ptr.clone(), offset + add),
            Val::Imm(_) => panic!("Offset from immediate"),
            Val::Void => panic!("Use of void value"),
        }
    }
}

pub struct Gen {
    // Current function
    frame_size: usize,
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

    pub fn do_sym(&mut self, vis: Vis, name: Rc<str>, dtype: Type) {
        self.symtab.insert(name.clone(), Sym::make_global(dtype, vis))
    }

    pub fn do_string(&mut self, suffix: Type, data: &str) -> Rc<str> {
        // Create assembly symbol
        let name: Rc<str> = format!("str${}", self.str_no).into();
        self.str_no += 1;
        // Deduce real type
        let dtype = if let Type::Deduce = suffix {
            Type::U8
        } else {
            suffix
        };
        // Generate data
        write!(self.rodata, "{} {} ", name, asm_dataword(&dtype)).unwrap();
        for byte in data.bytes() {
            write!(self.rodata, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(self.rodata, "0").unwrap();
        // Insert symbol
        self.do_sym(Vis::Private, name.clone(), Type::Array {
            elem_type: Box::new(dtype),
            elem_count: data.len()
        });
        name
    }

    fn gen_static_init(&mut self, dtype: &Type, init: Init) {
        match dtype {
            Type::U8|Type::I8|Type::U16|Type::I16|Type::U32|Type::I32|Type::U64|Type::I64 => {
                // Integers can only be initialized by constant expressions
                writeln!(self.data, "{} {}", asm_dataword(dtype),
                    init.want_expr().want_const()).unwrap()
            },
            Type::Ptr {..} => {
                match init.want_expr() {
                    // Pointer initialized by a constant expression
                    Expr::Const(_, val) => writeln!(self.data, "dq {}", val).unwrap(),
                    // Pointer initialized by the addres of another symbol
                    Expr::Ref(sym) => {
                        if let Expr::Sym(name) = *sym {
                            writeln!(self.data, "dq {}", name).unwrap()
                        } else {
                            panic!("Non-constant global initializer")
                        }
                    },
                    _ => panic!("Non-constant global initializer")
                }
            },
            Type::Array { elem_type, elem_count } => {
                let init_list = init.want_list();
                let mut elems_left = *elem_count;
                for elem in init_list {
                    if elems_left == 0 {
                        panic!("Too many initializers for array")
                    }
                    self.gen_static_init(elem_type, elem);
                    elems_left -= 1;
                }
                if elems_left > 0 {
                    panic!("Too few initializers for array")
                }
            },
            _ => todo!(),
        }
    }

    pub fn do_static_init(&mut self, vis: Vis, name: Rc<str>, dtype: Type, init: Init) {
        // Generate heading
        writeln!(self.data, "{}:", name).unwrap();
        // Generate data
        // FIXME: align .data entry
        self.gen_static_init(&dtype, init);
        // Create symbol
        self.do_sym(vis, name, dtype);
    }

    pub fn do_static(&mut self, vis: Vis, name: Rc<str>, dtype: Type) {
        // Allocate bss entry
        // FIXME: align .bss entry
        writeln!(self.bss, "{} resb {}", name, dtype.get_size()).unwrap();
        // Create symbol
        self.do_sym(vis, name, dtype);
    }

    fn stack_alloc(&mut self, dtype: &Type) -> usize {
        // FIXME: align allocation
        let offset = self.frame_size;
        self.frame_size += dtype.get_size();
        offset
    }

    fn alloc_temporary(&mut self, dtype: &Type) -> Val {
        let offset = self.stack_alloc(dtype);
        Val::Off(offset)
    }

    fn gen_unary(&mut self, op: &str, expr: Expr) -> (Type, Val) {
        let (expr_type, expr_val) = self.gen_expr(expr);
        expr_val.val_to_reg(&mut self.code, &expr_type, Reg::Rax);
        writeln!(self.code, "{} {}", op, Reg::Rax.to_str(&expr_type)).unwrap();
        let result = self.alloc_temporary(&expr_type);
        result.set_from_reg(&mut self.code, &expr_type, Reg::Rax, Reg::Rbx);
        (expr_type, result)
    }

    fn gen_binary(&mut self, op: &str, lhs: Expr, rhs: Expr) -> (Type, Val) {
        // Evaluate operands
        let (lhs_type, lhs_val) = self.gen_expr(lhs);
        let (rhs_type, rhs_val) = self.gen_expr(rhs);
        let result_type = Type::do_deduce(lhs_type, rhs_type);

        lhs_val.val_to_reg(&mut self.code, &result_type, Reg::Rax);
        rhs_val.val_to_reg(&mut self.code, &result_type, Reg::Rbx);

        // Do operation
        writeln!(self.code, "{} {}, {}", op,
            Reg::Rax.to_str(&result_type),
            Reg::Rbx.to_str(&result_type)).unwrap();
        // Save result to temporary
        let result = self.alloc_temporary(&result_type);
        result.set_from_reg(&mut self.code, &result_type, Reg::Rax, Reg::Rbx);
        (result_type, result)
    }

    fn gen_shift(&mut self, op: &str, lhs: Expr, rhs: Expr) -> (Type, Val) {
        // Evaluate operands
        let (lhs_type, lhs_val) = self.gen_expr(lhs);
        let (rhs_type, rhs_val) = self.gen_expr(rhs);
        let result_type = Type::do_deduce(lhs_type, rhs_type);

        lhs_val.val_to_reg(&mut self.code, &result_type, Reg::Rax);
        rhs_val.val_to_reg(&mut self.code, &result_type, Reg::Rcx);

        // Do operation
        writeln!(self.code, "{} {}, cl", op, Reg::Rax.to_str(&result_type)).unwrap();
        // Save result to temporary
        let result = self.alloc_temporary(&result_type);
        result.set_from_reg(&mut self.code, &result_type, Reg::Rax, Reg::Rbx);
        (result_type, result)
    }

    fn gen_expr(&mut self, expr: Expr) -> (Type, Val) {
        match expr {
            // Constant value
            Expr::Const(dtype, val) => (dtype, Val::Imm(val)),
            // Reference to symbol
            Expr::Sym(name) => {
                let sym = self.symtab.lookup(&name);
                match sym.kind {
                    SymKind::Global(_)
                        => (sym.dtype.clone(), Val::Sym(name, 0)),
                    SymKind::Local(offset)
                        => (sym.dtype.clone(), Val::Off(offset)),
                }
            },

            // Pointer ref/deref
            Expr::Ref(base) => {
                // Save address to rax
                let (base_type, base_val) = self.gen_expr(*base);
                base_val.ptr_to_reg(&mut self.code, Reg::Rax);
                // Save pointer to temporary
                let ptr_type = Type::Ptr { base_type: Box::new(base_type) };
                let ptr_val = self.alloc_temporary(&ptr_type);
                ptr_val.set_from_reg(&mut self.code, &ptr_type, Reg::Rax, Reg::Rbx);
                (ptr_type, ptr_val)
            },
            Expr::Deref(ptr) => {
                let (ptr_type, ptr_val) = self.gen_expr(*ptr);
                if let Type::Ptr { base_type } = ptr_type {
                    (*base_type, Val::Deref(Box::new(ptr_val), 0))
                } else {
                    panic!("De-referenced non-pointer type")
                }
            },

            // Unary operations
            Expr::Inv(expr)
                => self.gen_unary("not", *expr),
            Expr::Neg(expr)
                => self.gen_unary("neg", *expr),

            // Postfix expressions
            Expr::Field(record, field) => {
                let (record_type, record_val) = self.gen_expr(*record);
                let (dtype, offset) = if let Type::Record
                        { fields, lookup, .. } = record_type {
                    let idx = lookup.get(&field)
                        .expect(&format!("Non-existent record field {}", field));
                    fields[*idx].clone()
                } else {
                    panic!("Field access of non-record type")
                };
                (dtype, record_val.with_offset(offset))
            },
            Expr::Elem(array, index) => {
                // Generate array
                let (array_type, array_val) = self.gen_expr(*array);
                let (elem_type, elem_count) = if let Type::Array { elem_type, elem_count } = array_type {
                    (*elem_type, elem_count)
                } else {
                    panic!("Indexed non-array type")
                };
                // Generate index
                let (index_type, index_val) = self.gen_expr(*index);

                if let Val::Imm(val) = index_val {
                    if val >= elem_count {
                        panic!("Out of bounds array index")
                    }
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
                    let ptr_type = Type::Ptr {
                        base_type: Box::new(elem_type.clone()) };
                    let ptr_val = self.alloc_temporary(&ptr_type);
                    ptr_val.set_from_reg(&mut self.code,
                            &ptr_type, Reg::Rax, Reg::Rbx);
                    (elem_type, Val::Deref(Box::new(ptr_val), 0))
                }
            },
            Expr::Call(func, args) => {
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
                let (varargs, rettype) = if let Type::Func
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

                if let Type::Void = rettype {
                    // Create unusable value
                    (rettype, Val::Void)
                } else {
                    // Move return value to temporary
                    let ret_val = self.alloc_temporary(&rettype);
                    ret_val.set_from_reg(&mut self.code, &rettype, Reg::Rax, Reg::Rbx);
                    (rettype, ret_val)
                }
            },

            // Binary operations
            Expr::Add(lhs, rhs)
                => self.gen_binary("add", *lhs, *rhs),
            Expr::Sub(lhs, rhs)
                => self.gen_binary("sub", *lhs, *rhs),
            Expr::Mul(lhs, rhs)
                => self.gen_binary("imul", *lhs, *rhs),
            Expr::Div(lhs, rhs) => todo!(),
            Expr::Rem(lhs, rhs) => todo!(),
            Expr::Or(lhs, rhs)
                => self.gen_binary("or", *lhs, *rhs),
            Expr::And(lhs, rhs)
                => self.gen_binary("and", *lhs, *rhs),
            Expr::Xor(lhs, rhs)
                => self.gen_binary("xor", *lhs, *rhs),
            Expr::Lsh(lhs, rhs)
                => self.gen_shift("shl", *lhs, *rhs),
            Expr::Rsh(lhs, rhs)
                => self.gen_shift("shr", *lhs, *rhs),

            // Cast
            Expr::Cast(expr, dtype) => {
                // FIXME: integer casts cannot be done this way
                let (_, val) = self.gen_expr(*expr);
                (dtype, val)
            },
        }
    }

    fn gen_local_init(&mut self, dest_type: Type, dest_val: Val, init: Init) {
        match dest_type {
            Type::U8|Type::I8|Type::U16|Type::I16|
                    Type::U32|Type::I32|Type::U64|Type::I64|Type::Ptr {..} => {
                let (src_type, src_val) = self.gen_expr(init.want_expr());
                let comp_type = Type::do_deduce(dest_type, src_type);
                src_val.val_to_reg(&mut self.code, &comp_type, Reg::Rax);
                dest_val.set_from_reg(&mut self.code, &comp_type, Reg::Rax, Reg::Rbx);
            },
            Type::Array { elem_type, elem_count } => {
                let init_list = init.want_list();

                if init_list.len() != elem_count {
                    panic!("Array initializer with wrong number of elements");
                }

                for (i, elem) in init_list.into_iter().enumerate() {
                    self.gen_local_init(*elem_type.clone(),
                        dest_val.with_offset(i * elem_type.get_size()),
                        elem);
                }
            },
            _ => todo!(),
        }
    }

    fn gen_jcc(&mut self, jmp_op: &str, label: &str, lhs: Expr, rhs: Expr) {
        let (lhs_type, lhs_val) = self.gen_expr(lhs);
        let (rhs_type, rhs_val) = self.gen_expr(rhs);
        let comp_type = Type::do_deduce(lhs_type, rhs_type);
        lhs_val.val_to_reg(&mut self.code, &comp_type, Reg::Rax);
        rhs_val.val_to_reg(&mut self.code, &comp_type, Reg::Rbx);
        writeln!(self.code, "cmp {}, {}\n{} .{}",
            Reg::Rax.to_str(&comp_type), Reg::Rbx.to_str(&comp_type),
            jmp_op, label).unwrap();
    }

    pub fn do_func(&mut self, name: Rc<str>, rettype: Type, param_tab: Vec<(Rc<str>, Type)>, stmts: Vec<Stmt>) {
        self.frame_size = 0;
        self.code.clear();

        // Generate heading
        writeln!(self.text, "{}:", name).unwrap();

        // Create function scope with parameters
        self.symtab.push_scope();
        for (i, (name, dtype)) in param_tab.into_iter().enumerate() {
            let offset = self.stack_alloc(&dtype);
            Val::Off(offset)
                .set_from_reg(&mut self.code, &dtype, PARAMS[i], Reg::Rbx);
            self.symtab.insert(name, Sym::make_local(dtype, offset));
        }

        // Generate statements
        for stmt in stmts.into_iter() {
            match stmt {
                Stmt::Eval(expr) => {
                    self.gen_expr(expr);
                },
                Stmt::Ret(opt_expr) => {
                    // Evaluate return value if present
                    if let Some(expr) = opt_expr {
                        let (val_type, val) = self.gen_expr(expr);
                        let comp_type = Type::do_deduce(val_type, rettype.clone());
                        val.val_to_reg(&mut self.code, &comp_type, Reg::Rax);
                    }
                    // Then jump to the end of function
                    writeln!(&mut self.code, "jmp .$done").unwrap();
                },
                Stmt::Auto(name, dtype, opt_init) => {
                    // Allocate local variable
                    let offset = self.stack_alloc(&dtype);
                    // Generate initializer if provided
                    if let Some(init) = opt_init {
                        self.gen_local_init(dtype.clone(), Val::Off(offset), init);
                    }
                    // Create symbol for it
                    self.symtab.insert(name, Sym::make_local(dtype, offset));
                },
                Stmt::Label(label)
                    => writeln!(&mut self.code, ".{}:", label).unwrap(),
                Stmt::Set(dest, src) => {
                    let (dest_type, dest_val) = self.gen_expr(dest);
                    let (src_type, src_val) = self.gen_expr(src);
                    let comp_type = Type::do_deduce(src_type, dest_type);
                    src_val.val_to_reg(&mut self.code, &comp_type, Reg::Rax);
                    dest_val.set_from_reg(&mut self.code, &comp_type, Reg::Rax, Reg::Rbx);
                },
                Stmt::Jmp(label)
                    => writeln!(&mut self.code, "jmp .{}", label).unwrap(),
                Stmt::Jeq(label, lhs, rhs)
                    => self.gen_jcc("je", &*label, lhs, rhs),
                Stmt::Jneq(label, lhs, rhs)
                    => self.gen_jcc("jne", &*label, lhs, rhs),
                Stmt::Jl(label, lhs, rhs)
                    => self.gen_jcc("jl", &*label, lhs, rhs),
                Stmt::Jle(label, lhs, rhs)
                    => self.gen_jcc("jle", &*label, lhs, rhs),
                Stmt::Jg(label, lhs, rhs)
                    => self.gen_jcc("jg", &*label, lhs, rhs),
                Stmt::Jge(label, lhs, rhs)
                    => self.gen_jcc("jge", &*label, lhs, rhs),
            }
        }

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
