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
    R10 = 8,
    R11 = 9,
    R12 = 10,
    R13 = 11,
    R14 = 12,
    R15 = 13,
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
            _ => unreachable!(),
        }
    }
}

const ACCUM: Reg = Reg::Rax;
const PARAMS: [Reg; 6] = [ Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9 ];

enum SymKind {
    Global(Vis),
    Param(usize),
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

    fn make_param(dtype: Type, idx: usize) -> Sym {
        Sym {
            dtype: dtype,
            kind: SymKind::Param(idx),
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

enum Val {
    Imm(Type, usize),           // Immediate constant
    Off(Type, usize),           // Reference to stack
    Sym(Type, Rc<str>, usize),  // Reference to symbol
}

impl Val {
    fn get_type(self) -> Type {
        match self {
            Val::Imm(dtype, _) => dtype,
            Val::Off(dtype, _) => dtype,
            Val::Sym(dtype, _, _) => dtype,
        }
    }

    fn ref_type(&self) -> &Type {
        match self {
            Val::Imm(dtype, _) => &dtype,
            Val::Off(dtype, _) => &dtype,
            Val::Sym(dtype, _, _) => &dtype,
        }
    }

    fn ptr_to_reg(self, text: &mut String, reg: Reg) -> Type {
        match self {
            Val::Imm(_, _) => panic!("Cannot take address of immediate"),
            Val::Off(dtype, offset) => {
                let ptr_type = Type::Ptr { base_type: Box::new(dtype) };
                writeln!(text, "lea {}, [rsp + {}]",
                    reg.to_str(&ptr_type), offset).unwrap();
                ptr_type
            },
            Val::Sym(dtype, name, offset) => {
                let ptr_type = Type::Ptr { base_type: Box::new(dtype) };
                writeln!(text, "lea {}, [{} + {}]",
                    reg.to_str(&ptr_type), name, offset).unwrap();
                ptr_type
            },
        }
    }

    fn val_to_reg(&self, text: &mut String, reg: Reg) {
        match self {
            Val::Imm(dtype, val)
                => writeln!(text, "mov {}, {}", reg.to_str(dtype), val).unwrap(),
            Val::Off(dtype, offset)
                => writeln!(text, "mov {}, [rsp + {}]", reg.to_str(dtype), offset).unwrap(),
            Val::Sym(dtype, name, offset)
                => writeln!(text, "mov {}, [{} + {}]", reg.to_str(dtype), name, offset).unwrap(),
        }
    }

    fn set_from_reg(&self, text: &mut String, reg: Reg) {
        match self {
            Val::Imm(dtype, val)
                => writeln!(text, "mov {}, {}", val, reg.to_str(dtype)).unwrap(),
            Val::Off(dtype, offset)
                => writeln!(text, "mov [rsp + {}], {}", offset, reg.to_str(dtype)).unwrap(),
            Val::Sym(dtype, name, offset)
                => writeln!(text, "mov [{} + {}], {}", name, offset, reg.to_str(dtype)).unwrap(),
        }
    }
}

pub struct Gen {
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
        // Generate data
        write!(self.rodata, "{} {} ", name, asm_dataword(&suffix)).unwrap();
        for byte in data.bytes() {
            write!(self.rodata, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(self.rodata, "0").unwrap();
        // Insert symbol
        self.do_sym(Vis::Private, name.clone(), Type::Array {
            elem_type: Box::new(suffix),
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

    fn gen_expr(&mut self, frame_size: &mut usize, code: &mut String, expr: Expr) -> Val {

        fn alloc_temporary(frame_size: &mut usize, dtype: Type) -> Val {
            // FIXME: align temporary
            let offset = *frame_size;
            *frame_size += dtype.get_size();
            Val::Off(dtype, offset)
        }

        match expr {
            // Constant value
            Expr::Const(dtype, val) => Val::Imm(dtype, val),
            // Reference to symbol
            Expr::Sym(name) => {
                let sym = self.symtab.lookup(&name);
                match sym.kind {
                    SymKind::Global(_)
                        => Val::Sym(sym.dtype.clone(), name, 0),
                    SymKind::Param(index)
                        => todo!(),
                    SymKind::Local(offset)
                        => Val::Off(sym.dtype.clone(), offset),
                }
            },
            // Pointer ref/deref
            Expr::Ref(base) => {
                // Save address to rax
                let base_val = self.gen_expr(frame_size, code, *base);
                let ptr_type = base_val.ptr_to_reg(code, ACCUM);
                // Save pointer to temporary
                let ptr_val = alloc_temporary(frame_size, ptr_type);
                ptr_val.set_from_reg(code, ACCUM);
                ptr_val
            },
            Expr::Deref(ptr) => {
                // Move pointer value to rax
                let ptr_val = self.gen_expr(frame_size, code, *ptr);
                ptr_val.val_to_reg(code, ACCUM);
                let ptr_type = ptr_val.get_type();
                let ptr_reg = ACCUM.to_str(&ptr_type);
                // Read from pointer
                if let Type::Ptr { base_type } = ptr_type {
                    // Dereference pointer in rax
                    writeln!(code, "mov {}, [{}]",
                        ACCUM.to_str(&*base_type), ptr_reg).unwrap();
                    // Write dereferenced value to temporary
                    let base_val = alloc_temporary(frame_size, *base_type);
                    base_val.set_from_reg(code, ACCUM);
                    base_val
                } else {
                    panic!("Derefernced non-pointer type")
                }
            },
            // Unary operations
            Expr::Inv(expr) => todo!(),
            Expr::Neg(expr) => todo!(),
            // Postfix expressions
            Expr::Field(expr, name) => todo!(),
            Expr::Elem(expr, index) => todo!(),
            Expr::Call(func, args) => {
                // FIXME: this is very ugly
                let (name, varargs, rettype) = if let Expr::Sym(name) = *func {
                    let sym = self.symtab.lookup(&name);
                    if let SymKind::Global(_) = sym.kind {
                        if let Type::Func { params, varargs, rettype } = &sym.dtype {
                            (name, *varargs, *rettype.clone())
                        } else {
                            panic!("Non callable object called")
                        }
                    } else {
                        panic!("Non callable object called")
                    }
                } else {
                    panic!("Non callable object called")
                };

                // Move arguments to registers
                // FIXME: more than 6 arguments
                for (arg, reg) in args.into_iter().zip(PARAMS) {
                    let arg_val = self.gen_expr(frame_size, code, arg);
                    arg_val.val_to_reg(code, reg);
                }

                // Generate call
                if varargs {
                    writeln!(code, "xor eax, eax").unwrap();
                }
                writeln!(code, "call {}", name).unwrap();

                // Move return value to temporary
                let ret_val = alloc_temporary(frame_size, rettype);
                ret_val.set_from_reg(code, Reg::Rax);
                ret_val
            },
            // Binary operations
            Expr::Add(lhs, rhs) => {
                // Evaluate operands
                let lhs_val = self.gen_expr(frame_size, code, *lhs);
                lhs_val.val_to_reg(code, Reg::Rax);
                let rhs_val = self.gen_expr(frame_size, code, *rhs);
                rhs_val.val_to_reg(code, Reg::Rbx);
                // Do operation
                let result_type = lhs_val.get_type();
                writeln!(code, "add {}, {}",
                    Reg::Rax.to_str(&result_type),
                    Reg::Rbx.to_str(&result_type)).unwrap();
                // Save result to temporary
                let result = alloc_temporary(frame_size, result_type);
                result.set_from_reg(code, Reg::Rax);
                result
            },
            Expr::Sub(lhs, rhs) => {
                // Evaluate operands
                let lhs_val = self.gen_expr(frame_size, code, *lhs);
                lhs_val.val_to_reg(code, Reg::Rax);
                let rhs_val = self.gen_expr(frame_size, code, *rhs);
                rhs_val.val_to_reg(code, Reg::Rbx);
                // Do operation
                let result_type = lhs_val.get_type();
                writeln!(code, "sub {}, {}",
                    Reg::Rax.to_str(&result_type),
                    Reg::Rbx.to_str(&result_type)).unwrap();
                // Save result to temporary
                let result = alloc_temporary(frame_size, result_type);
                result.set_from_reg(code, Reg::Rax);
                result
            },
            Expr::Mul(lhs, rhs) => todo!(),
            Expr::Div(lhs, rhs) => todo!(),
            Expr::Rem(lhs, rhs) => todo!(),
            Expr::Or(lhs, rhs) => todo!(),
            Expr::And(lhs, rhs) => todo!(),
            Expr::Xor(lhs, rhs) => todo!(),
            Expr::Lsh(lhs, rhs) => todo!(),
            Expr::Rsh(lhs, rhs) => todo!(),
            // Cast
            Expr::Cast(expr, dtype) => todo!(),
        }
    }

    fn gen_local_init(&mut self, frame_size: &mut usize, code: &mut String,
                        dest_val: &Val, init: Init) {
        match dest_val.ref_type() {
            Type::U8|Type::I8|Type::U16|Type::I16|
                    Type::U32|Type::I32|Type::U64|
                    Type::I64|Type::Ptr {..} => {
                let src_val = self.gen_expr(frame_size, code, init.want_expr());
                src_val.val_to_reg(code, Reg::Rax);
                dest_val.set_from_reg(code, Reg::Rax);
            },
            _ => todo!(),
        }
    }

    pub fn do_func(&mut self, name: Rc<str>, param_tab: Vec<(Rc<str>, Type)>, stmts: Vec<Stmt>) {
        // Generate heading
        writeln!(self.text, "{}:", name).unwrap();

        // Create function scope with parameters
        self.symtab.push_scope();
        for (i, (name, dtype)) in param_tab.into_iter().enumerate() {
            self.symtab.insert(name, Sym::make_param(dtype, i));
        }

        // We store the stackframe size here and pass around a reference to it
        // This will be adjusted when allocating local variabels or temporaries
        let mut frame_size = 0;

        // Generate statements
        let mut code = String::new();
        for stmt in stmts.into_iter() {
            match stmt {
                Stmt::Eval(expr) => {
                    self.gen_expr(&mut frame_size, &mut code, expr);
                },
                Stmt::Ret(opt_expr) => {
                    // Evaluate return value if present
                    if let Some(expr) = opt_expr {
                        let expr_val = self.gen_expr(&mut frame_size, &mut code, expr);
                        expr_val.val_to_reg(&mut code, Reg::Rax);
                    }
                    // Then jump to the end of function
                    writeln!(code, "jmp .$done").unwrap();
                },
                Stmt::Auto(name, dtype, opt_init) => {
                    // Allocate local variable
                    // FIXME: align stack frame
                    let offset = frame_size;
                    frame_size += dtype.get_size();

                    // Generate initializer if provided
                    if let Some(init) = opt_init {
                        self.gen_local_init(&mut frame_size, &mut code,
                            &Val::Off(dtype.clone(), offset), init);
                    }

                    // Create symbol for it
                    self.symtab.insert(name, Sym::make_local(dtype, offset));
                },
                Stmt::Label(label)
                    => writeln!(code, ".{}:", label).unwrap(),
                Stmt::Set(dest, src) => {
                    let dest_val = self.gen_expr(&mut frame_size, &mut code, dest);
                    let src_val = self.gen_expr(&mut frame_size, &mut code, src);
                    src_val.val_to_reg(&mut code, Reg::Rax);
                    dest_val.set_from_reg(&mut code, Reg::Rax);
                },
                Stmt::Jmp(label)
                    => writeln!(code, "jmp .{}", label).unwrap(),
                Stmt::Jeq(label, expr1, expr2) => {
                },
                Stmt::Jneq(label, expr1, expr2) => {
                },
                Stmt::Jl(label, expr1, expr2) => {
                },
                Stmt::Jle(label, expr1, expr2) => {
                },
                Stmt::Jg(label, expr1, expr2) => {
                },
                Stmt::Jge(label, expr1, expr2) => {
                },
            }
        }

        // Round stack frame
        frame_size = (frame_size + 15) / 16 * 16 + 8;

        // Generate code
        writeln!(self.text, "push rbp\nmov rbp, rsp\nsub rsp, {}\n{}.$done:\nleave\nret",
            frame_size, code).unwrap();

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
