// SPDX-License-Identifier: GPL-2.0-only

//
// AST to IL conversion
//

use super::ast;
use std::collections::HashMap;
use std::rc::Rc;

// Local variable
#[derive(Debug)]
struct Local {
    used: bool,     // Ever used?
    refed: bool,    // Address taken?
    align: usize,   // Required alingment
    size: usize,    // Required size
}

impl Local {
    fn new(dtype: &ast::Type) -> Local {
        Local {
            used: false,
            refed: false,
            // FIXME: implement proper alingment
            align: dtype.get_size(),
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

impl Operand {
    fn width(&self) -> Width {
        match self {
            Operand::Byte(_) | Operand::ByteReg(_) => Width::Byte,
            Operand::Word(_) | Operand::WordReg(_) => Width::Word,
            Operand::DWord(_) | Operand::DWordReg(_) => Width::DWord,
            Operand::QWord(_) | Operand::QWordReg(_) => Width::QWord,
            Operand::Ptr(_) | Operand::PtrReg(_) |
                Operand::LocalPtr(_) | Operand::StaticPtr(_) => Width::Ptr,
        }
    }
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
    Call(Rc<str>, Operand, Vec<Operand>),
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
            if let ast::Type::Record(record) = &*dtype {
                if let Some((field_type, offset)) = record.fields.get(ident) {
                    let reg = alloc_reg(Width::Ptr);
                    ops.push(Op::Set(reg.clone(), lval.to_ptr()));
                    ops.push(Op::Add(reg.clone(), Operand::Ptr(*offset as u64)));
                    (field_type.clone(), LVal::Deref(reg))
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
                ops.push(Op::Mul(idx.clone(), Operand::Ptr(elem_type.get_size() as u64)));
                ops.push(Op::Add(idx.clone(), lval.to_ptr()));
                (elem_type.clone(), LVal::Deref(idx))
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
                let ret_operand = alloc_reg(load_width(&func.rettype));
                let mut param_operands = Vec::new();
                for param in params {
                    // FIXME: check parameter type
                    let (_, operand) = conv_expr(param, res, ops);
                    param_operands.push(operand);
                }
                ops.push(Op::Call(ident.clone(), ret_operand.clone(), param_operands));
                (func.rettype.clone(), ret_operand)
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
        let mut res = Resolv {
            file: ast_file,
            locals: HashMap::new(),
        };
        let mut ops = Vec::new();

        // Create locals for parameters
        for (name, dtype) in &ast_func.params {
            res.locals.insert(name, (dtype.clone(), Rc::from(Local::new(dtype))));
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

                ast::Stmt::Auto(name, dtype, init) => {
                    // Make sure the identifier is not already used
                    if let Some(_) = res.locals.get(name) {
                        panic!("Identifier {} already in use in the same scope", name)
                    }
                    // Allocate local
                    res.locals.insert(name, (dtype.clone(), Rc::from(Local::new(dtype))));
                    // TODO: Generate initializer
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

                _ => todo!("statement {:?}", stmt),
            }
        }

        Func {
            locals: res.locals.drain().map(|(_, (_, local))| local).collect(),
            ops: ops
        }
    }
}

