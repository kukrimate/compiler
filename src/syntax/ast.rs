// SPDX-License-Identifier: GPL-2.0-only

use crate::gen::Local;
use std::collections::HashMap;
use std::rc::Rc;

//
// Type expressions
//

#[derive(Clone,Debug)]
pub enum Ty {
    Var(usize),

    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    USize,

    Bool,

    Ptr {
        // What type does this point to?
        base_type: Box<Ty>,
    },

    Array {
        // Ty of array elements
        elem_type: Box<Ty>,
        // Number of array elements
        elem_count: Option<usize>,
    },

    // Product type
    Record {
        // Name of the record (this must match during type checking)
        name: Rc<str>,
        // Name lookup table
        lookup: HashMap<Rc<str>, usize>,
        // Field types and offsets (in declaration order)
        fields: Box<[(Ty, usize)]>,
        // Pre-calculated alingment and size
        align: usize,
        size: usize,
    },

    // Return type from a procedure
    Void,

    // Function
    Func {
        params: Box<[Ty]>,
        varargs: bool,
        rettype: Box<Ty>,
    },
}

impl Ty {
    pub fn is_signed(&self) -> bool {
        match self {
            Ty::I8|Ty::I16|Ty::I32|Ty::I64 => true,
            _ => false
        }
    }

    pub fn get_align(&self) -> usize {
        match self {
            Ty::U8 | Ty::I8 | Ty::Bool => 1,
            Ty::U16 | Ty::I16 => 2,
            Ty::U32 | Ty::I32 => 4,
            Ty::U64 | Ty::I64 | Ty::USize | Ty::Ptr {..} => 8,
            Ty::Array { elem_type, .. } => elem_type.get_align(),
            Ty::Record { align, .. } => *align,
            Ty::Var(_) | Ty::Void | Ty::Func {..} => unreachable!(),
        }
    }

    pub fn get_size(&self) -> usize {
        match self {
            Ty::U8 | Ty::I8 | Ty::Bool => 1,
            Ty::U16 | Ty::I16 => 2,
            Ty::U32 | Ty::I32 => 4,
            Ty::U64 | Ty::I64 | Ty::USize | Ty::Ptr {..} => 8,
            Ty::Array { elem_type, elem_count } => elem_type.get_size() * elem_count.expect("Array without element count allocated"),
            Ty::Record { size, .. } => *size,
            Ty::Var(_) | Ty::Void | Ty::Func {..} => unreachable!(),
        }
    }
}


//
// Expressions
//

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum UOp {
    Not,
    Neg,
}

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum BOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Lsh,
    Rsh,
    And,
    Xor,
    Or,
}

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Cond {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Clone,Debug)]
pub enum ExprKind {
    // Constant value
    Const(usize),
    // Array/Record literal
    Compound(Vec<Expr>),
    // Reference to symbol
    Global(Rc<str>),
    Local(Rc<Local>),
    // Postfix expressions
    Field(Box<Expr>, usize),
    Call(Box<Expr>, Vec<Expr>, bool),
    Elem(Box<Expr>, Box<Expr>),

    // Indirection
    Ref(Box<Expr>),
    Deref(Box<Expr>),

    // Unary arithmetic
    Unary(UOp, Box<Expr>),

    // Cast
    Cast(Box<Expr>),

    // Binary arithmetic
    Binary(BOp, Box<Expr>, Box<Expr>),

    // Conditional expressions
    Cond(Cond, Box<Expr>, Box<Expr>),

    // Boolean expressions
    LNot(Box<Expr>),
    LAnd(Box<Expr>, Box<Expr>),
    LOr(Box<Expr>, Box<Expr>),
}

#[derive(Clone,Debug)]
pub struct Expr {
    pub ty: Ty,
    pub kind: ExprKind,
}

impl Expr {
    pub fn make_const(ty: Ty, val: usize) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Const(val)
        }
    }

    pub fn make_global(ty: Ty, name: Rc<str>) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Global(name),
        }
    }

    pub fn make_local(ty: Ty, local: Rc<Local>) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Local(local),
        }
    }
}

#[derive(Debug)]
pub enum Stmt {
    Block(Vec<Stmt>),
    Eval(Expr),
    Ret(Option<Expr>),
    Auto(Ty, Rc<Local>, Option<Expr>),
    Label(Rc<str>),
    Set(Expr, Expr),
    Jmp(Rc<str>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
}
