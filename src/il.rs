// SPDX-License-Identifier: GPL-2.0-only

//
// AST to IL conversion
//

use crate::ast;
use std::collections::HashMap;
use std::rc::Rc;

// Local variable
#[derive(Debug)]
struct Local {
    name: Rc<str>,  // Variable name
    size: usize,    // Required size
}

impl Local {
    fn new(name: Rc<str>, dtype: &ast::Type) -> Local {
        Local {
            name: name,
            size: dtype.get_size(),
        }
    }
}

#[derive(Debug)]
enum Width {
    Byte,
    Word,
    DWord,
    QWord,
    Ptr
}

// Width of a type when loaded
fn load_width(dtype: &ast::Type) -> Width {
    match dtype {
        ast::Type::U8 => Width::Byte,
        ast::Type::I8 => Width::Byte,
        ast::Type::U16 => Width::Word,
        ast::Type::I16 => Width::Word,
        ast::Type::U32 => Width::DWord,
        ast::Type::I32 => Width::DWord,
        ast::Type::U64 => Width::QWord,
        ast::Type::I64 => Width::QWord,
        ast::Type::Ptr{..} => Width::Ptr,
        _ => panic!("Type {:?} cannot be used as an rvalue", dtype)
    }
}

#[derive(Debug,Clone)]
enum Operand {
    // Constants
    Byte(u8),
    Word(u16),
    DWord(u32),
    QWord(u64),
    Ptr(u64),
    // Temporaries
    ByteReg(usize),
    WordReg(usize),
    DWordReg(usize),
    QWordReg(usize),
    PtrReg(usize),
    // Symbol pointers
    LocalPtr(Rc<Local>),
    StaticPtr(Rc<str>),
}

#[derive(Debug)]
enum Op {
    // Copy value
    Set(Operand, Operand),
    // Read and write memory
    Load(Operand, Operand),
    Store(Operand, Operand),
    // Unary operations
    Inv(Operand),
    Neg(Operand),
    // Binary operations
    Add(Operand, Operand),
    Sub(Operand, Operand),
    Mul(Operand, Operand),
    Div(Operand, Operand),
    Rem(Operand, Operand),
    And(Operand, Operand),
    Or(Operand, Operand),
    Xor(Operand, Operand),
    Lsh(Operand, Operand),
    Rsh(Operand, Operand),
    // Function Call
    Call(Rc<str>, Vec<Operand>, Operand),
    GotoSub(Rc<str>, Vec<Operand>),
    // Define a label
    Label(Rc<str>),
    // Jumps
    Jmp(Rc<str>),
    Je(Rc<str>, Operand, Operand),
    Jne(Rc<str>, Operand, Operand),
    Jl(Rc<str>, Operand, Operand),
    Jle(Rc<str>, Operand, Operand),
    Jg(Rc<str>, Operand, Operand),
    Jge(Rc<str>, Operand, Operand),
    // Return
    Return(Operand),
    GotoEnd,
}

static mut REGCNT: usize = 0;

fn alloc_reg(width: Width) -> Operand {
    unsafe {
        let regno = REGCNT;
        REGCNT += 1;

        match width {
            Width::Byte => Operand::ByteReg(regno),
            Width::Word => Operand::WordReg(regno),
            Width::DWord => Operand::DWordReg(regno),
            Width::QWord => Operand::QWordReg(regno),
            Width::Ptr => Operand::PtrReg(regno),
        }
    }
}


struct Resolv<'a> {
    file: &'a ast::File,
    locals: HashMap<&'a Rc<str>, (Rc<ast::Type>, Rc<Local>)>,
}

impl<'a> Resolv<'a> {
    fn new(file: &'a ast::File) -> Resolv {
        Resolv {
            file: file,
            locals: HashMap::new(),
        }
    }

    fn resolve_name(&self, name: &Rc<str>) -> (Rc<ast::Type>, LVal) {
        if let Some((dtype, local)) = self.locals.get(name) {
            (dtype.clone(), LVal::Local(local.clone()))
        } else if let Some(s) = self.file.statics.get(name) {
            (Rc::from(s.dtype.clone()), LVal::Static(name.clone()))
        } else {
            panic!("Unknown identifier {}", name)
        }
    }

    fn resolve_func(&self, name: &Rc<str>) -> &'a ast::Func {
        if let Some(func) = self.file.funcs.get(name) {
            func
        } else {
            panic!("Unknown function {}", name)
        }
    }
}

#[derive(Clone)]
enum LVal {
    // Reference to a symbol
    Local(Rc<Local>),
    Static(Rc<str>),
    // Dereference (of a pointer)
    Deref(Operand),
}

impl LVal {
    fn to_ptr(self) -> Operand {
        match self {
            LVal::Local(local) => Operand::LocalPtr(local),
            LVal::Static(name) => Operand::StaticPtr(name),
            LVal::Deref(ptr) => ptr,
        }
    }

    fn to_field(self, offset: usize, ops: &mut Vec<Op>) -> LVal {
        let reg = alloc_reg(Width::Ptr);
        ops.push(Op::Set(reg.clone(), self.to_ptr()));
        ops.push(Op::Add(reg.clone(), Operand::Ptr(offset as u64)));
        LVal::Deref(reg)
    }

    fn to_elem(self, idx: Operand, elem_size: usize, ops: &mut Vec<Op>) -> LVal {
        ops.push(Op::Mul(idx.clone(), Operand::Ptr(elem_size as u64)));
        ops.push(Op::Add(idx.clone(), self.to_ptr()));
        LVal::Deref(idx)
    }
}

fn conv_lval<'a>(expr: &'a ast::Expr, res: &'a Resolv, ops: &mut Vec<Op>) -> (Rc<ast::Type>, LVal) {
    match expr {
        ast::Expr::Ident(ident) => res.resolve_name(&ident),

        ast::Expr::Deref(ptr_expr) => {
            let (dtype, operand) = conv_expr(ptr_expr, res, ops);
            if let ast::Type::Ptr { base_type } = &*dtype {
                (base_type.clone(), LVal::Deref(operand))
            } else {
                panic!("Cannot dereference non-pointer type")
            }
        },

        ast::Expr::Field(expr, ident) => {
            let (dtype, lval) = conv_lval(expr, res, ops);
            if let ast::Type::Record { fields, field_map, .. } = &*dtype {
                if let Some(i) = field_map.get(ident) {
                    let (field_type, field_offset) = &fields[*i];
                    (field_type.clone(), lval.to_field(*field_offset, ops))
                } else {
                    panic!("Record field {} doesn't exist", ident)
                }
            } else {
                panic!("Cannot access field of non-record type")
            }
        },

        ast::Expr::Elem(expr, idx_expr) => {
            let (dtype, lval) = conv_lval(expr, res, ops);
            // FIXME: make sure index is a scalar value
            let (_, idx) = conv_expr(idx_expr, res, ops);
            if let ast::Type::Array { elem_type, .. } = &*dtype {
                (elem_type.clone(), lval.to_elem(idx, elem_type.get_size(), ops))
            } else {
                panic!("Cannot access field of non-record type")
            }
        },

        _ => panic!("Expected lvalue expression"),
    }
}

fn conv_expr<'a>(expr: &'a ast::Expr, res: &'a Resolv, ops: &mut Vec<Op>) -> (Rc<ast::Type>, Operand) {
    macro_rules! conv_uop {
        ($op:path, $expr:expr) => {{
            let (dtype, operand) = conv_expr($expr, res, ops);
            ops.push($op(operand.clone()));
            (dtype, operand)
        }}
    }

    macro_rules! conv_bop {
        ($op:path, $lhs:expr, $rhs:expr) => {{
            let (ltype, loperand) = conv_expr($lhs, res, ops);
            let (rtype, roperand) = conv_expr($rhs, res, ops);
            if ltype != rtype {
                panic!("Binary operation on mis-matched types {:?} and {:?}", ltype, rtype)
            }
            ops.push($op(loperand.clone(), roperand));
            (ltype, loperand)
        }}
    }

    match expr {
        // Rvalue expressions (pass directly)
        ast::Expr::Const(dtype, val) => {
            match **dtype {
                ast::Type::U8 => (Rc::from(dtype.clone()), Operand::Byte(*val as u8)),
                ast::Type::I8 => (Rc::from(dtype.clone()), Operand::Byte(*val as u8)),
                ast::Type::U16 => (Rc::from(dtype.clone()), Operand::Word(*val as u16)),
                ast::Type::I16 => (Rc::from(dtype.clone()), Operand::Word(*val as u16)),
                ast::Type::U32 => (Rc::from(dtype.clone()), Operand::DWord(*val as u32)),
                ast::Type::I32 => (Rc::from(dtype.clone()), Operand::DWord(*val as u32)),
                ast::Type::U64 => (Rc::from(dtype.clone()), Operand::QWord(*val as u64)),
                ast::Type::I64 => (Rc::from(dtype.clone()), Operand::QWord(*val as u64)),
                ast::Type::Ptr{..} => (Rc::from(dtype.clone()), Operand::Ptr(*val as u64)),
                _ => panic!("Constant cannot have type {:?}", dtype)
            }
        },

        ast::Expr::Call(expr, params) => {
            if let ast::Expr::Ident(ident) = &**expr {
                let func = res.resolve_func(ident);

                let mut operands = Vec::new();
                for param in params {
                    // FIXME: check parameter type
                    let (_, operand) = conv_expr(param, res, ops);
                    operands.push(operand);
                }

                // Evaluate each parameter
                let operands = params.iter()
                    .map(|param: &ast::Expr| conv_expr(param, res, ops).1)
                    .collect();

                // Generate call operation
                if let Some(rettype) = func.rettype.as_ref() {
                    let reg = alloc_reg(load_width(rettype));
                    ops.push(Op::Call(ident.clone(), operands, reg.clone()));
                    (rettype.clone(), reg)
                } else {
                    todo!("implement gotosub")
                }
            } else {
                panic!("Call target must be an identifier");
            }
        },

        ast::Expr::Inv(expr) => conv_uop!(Op::Inv, expr),
        ast::Expr::Neg(expr) => conv_uop!(Op::Neg, expr),

        ast::Expr::Add(lhs, rhs) => conv_bop!(Op::Add, lhs, rhs),
        ast::Expr::Sub(lhs, rhs) => conv_bop!(Op::Sub, lhs, rhs),
        ast::Expr::Mul(lhs, rhs) => conv_bop!(Op::Mul, lhs, rhs),
        ast::Expr::Div(lhs, rhs) => conv_bop!(Op::Div, lhs, rhs),
        ast::Expr::Rem(lhs, rhs) => conv_bop!(Op::Rem, lhs, rhs),
        ast::Expr::Or(lhs, rhs)  => conv_bop!(Op::Or,  lhs, rhs),
        ast::Expr::And(lhs, rhs) => conv_bop!(Op::And, lhs, rhs),
        ast::Expr::Xor(lhs, rhs) => conv_bop!(Op::Xor, lhs, rhs),
        ast::Expr::Lsh(lhs, rhs) => conv_bop!(Op::Lsh, lhs, rhs),
        ast::Expr::Rsh(lhs, rhs) => conv_bop!(Op::Rsh, lhs, rhs),

        ast::Expr::Cast(expr, new_type) => {
            let (_, operand) = conv_expr(expr, res, ops);
            (new_type.clone(), operand)
        },

        // Compute address of lvalue expression
        ast::Expr::Ref(expr) => {
            // Get lvalue expression
            let (dtype, lval) = conv_lval(expr, res, ops);
            // Find the underlying pointer
            let ptr = match lval {
                LVal::Local(local) => Operand::LocalPtr(local),
                LVal::Static(name) => Operand::StaticPtr(name),
                LVal::Deref(ptr) => ptr,
            };
            // Create type for pointer and return it with the type
            (Rc::from(ast::Type::Ptr { base_type: dtype }), ptr)
        },


        // Compute value of lvalue expressions
        expr => {
            // Get lvalue with type
            let (dtype, lval) = conv_lval(expr, res, ops);
            // Do load from lvalue
            let reg_operand = alloc_reg(load_width(&dtype));
            ops.push(Op::Load(reg_operand.clone(), lval.to_ptr()));
            (dtype, reg_operand)
        },
    }
}

fn conv_init<'a>(dtype: &Rc<ast::Type>, dest: LVal, init: &ast::Init, res: &'a Resolv, ops: &mut Vec<Op>) {
    match &**dtype {
        ast::Type::Array { elem_type, elem_count } => {
            let init_list = init.want_list();
            if init_list.len() != *elem_count {
                panic!("Invalid array initializer!");
            }
            for (i, elem_init) in init_list.iter().enumerate() {
                let idx_operand = Operand::Ptr(i as u64);
                let elem_lval = dest.clone().to_elem(idx_operand, elem_type.get_size(), ops);
                conv_init(elem_type, elem_lval, elem_init, res, ops);
            }
        },

        ast::Type::Record { fields, .. } => {
            for ((field_type, field_offset), field_init) in fields.iter().zip(init.want_list()) {
                let field_lval = dest.clone().to_field(*field_offset, ops);
                conv_init(field_type, field_lval, field_init, res, ops);
            }
        },

        // Integer/pointer types (base initializer)
        dtype => {
            let (init_type, operand) = conv_expr(init.want_expr(), res, ops);
            if dtype != &*init_type {
                panic!("Initializer with the wrong type expected {:?}, got {:?}", dtype, init_type)
            }
            ops.push(Op::Store(dest.to_ptr(), operand));
        },
    }
}

// Function
#[derive(Debug)]
pub struct Func {
    // Local variables
    locals: Vec<Rc<Local>>,
    // Operations
    ops: Vec<Op>,
}

impl Func {
    // Create a new IL function from an AST function
    pub fn new(ast_file: &ast::File, ast_func: &ast::Func) -> Func {
        let mut res = Resolv::new(ast_file);
        let mut ops = Vec::new();

        // Create locals for parameters
        for (name, dtype) in &ast_func.params {
            res.locals.insert(name,
                (dtype.clone(), Rc::from(Local::new(name.clone(), dtype))));
        }

        macro_rules! conv_jcc {
            ($op:path,$label:expr,$expr1:expr,$expr2:expr) => {{
                let (type1, val1) = conv_expr($expr1, &mut res, &mut ops);
                let (type2, val2) = conv_expr($expr2, &mut res, &mut ops);
                if type1 != type2 {
                    panic!("Comparison type mis-match {:?} and {:?}", type1, type2)
                }
                ops.push($op($label.clone(), val1, val2));
            }}
        }

        // Translate to IL operations
        for stmt in &ast_func.stmts {
            match stmt {
                ast::Stmt::Eval(expr) => {
                    conv_expr(expr, &mut res, &mut ops);
                },

                ast::Stmt::Ret(maybe_expr) => {
                    // FIXME: validate return type
                    if let Some(expr) = maybe_expr {
                        let (_, operand) = conv_expr(expr, &mut res, &mut ops);
                        ops.push(Op::Return(operand));
                    } else {
                        ops.push(Op::GotoEnd);
                    }
                },

                ast::Stmt::Auto(name, dtype, maybe_init) => {
                    if let Some(_) = res.locals.get(name) {
                        panic!("Identifier {} already in use in the same scope", name)
                    }
                    let local = Rc::from(Local::new(name.clone(), dtype));
                    if let Some(init) = maybe_init {
                        conv_init(dtype, LVal::Local(local.clone()), init, &mut res, &mut ops);
                    }
                    res.locals.insert(name, (dtype.clone(), local.clone()));
                },

                ast::Stmt::Label(label) => ops.push(Op::Label(label.clone())),

                ast::Stmt::Set(dest, src) => {
                    let (dtype, lval) = conv_lval(dest, &mut res, &mut ops);
                    let (stype, operand) = conv_expr(src, &mut res, &mut ops);
                    if dtype != stype {
                        panic!("Assingment with mis-matched types {:?} = {:?}", dtype, stype)
                    }
                    ops.push(Op::Store(lval.to_ptr(), operand));
                },

                ast::Stmt::Jmp(label) => {
                    ops.push(Op::Jmp(label.clone()));
                },

                ast::Stmt::Jeq(label, expr1, expr2) => conv_jcc!(Op::Je, label, expr1, expr2),
                ast::Stmt::Jneq(label, expr1, expr2) => conv_jcc!(Op::Jne, label, expr1, expr2),
                ast::Stmt::Jl(label, expr1, expr2) => conv_jcc!(Op::Jl, label, expr1, expr2),
                ast::Stmt::Jle(label, expr1, expr2) => conv_jcc!(Op::Jle, label, expr1, expr2),
                ast::Stmt::Jg(label, expr1, expr2) => conv_jcc!(Op::Jg, label, expr1, expr2),
                ast::Stmt::Jge(label, expr1, expr2) => conv_jcc!(Op::Jge, label, expr1, expr2),
            }
        }

        Func {
            locals: res.locals.drain().map(|(_, (_, local))| local).collect(),
            ops: ops
        }
    }
}

