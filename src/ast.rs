// SPDX-License-Identifier: GPL-2.0-only

//
// Abstract syntax tree elements
//

use std::collections::HashMap;
use std::rc::Rc;

//
// Type system
//

#[derive(Debug)]
pub struct Record {
    pub fields: HashMap<Rc<str>, (Type, usize)>,
    pub size: usize,
}

#[derive(Clone,Debug)]
pub enum Type {
    VOID,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    Ptr {
        base_type: Box<Type>,
    },
    Array {
        elem_type: Box<Type>,
        elem_count: usize,
    },
    Record(Rc<Record>),
}

pub const PTR_SIZE: usize = 8;

impl Type {
    pub fn get_size(&self) -> usize {
        match self {
            Type::VOID => unreachable!(),
            Type::U8  => 1,
            Type::I8  => 1,
            Type::U16 => 2,
            Type::I16 => 2,
            Type::U32 => 4,
            Type::I32 => 4,
            Type::U64 => 8,
            Type::I64 => 8,
            Type::Ptr {..} => PTR_SIZE,
            Type::Array { elem_type, elem_count } => {
                elem_type.get_size() * elem_count
            },
            Type::Record(record) => record.size,
        }
    }
}

//
// Functions
//

#[derive(Debug)]
pub enum Expr {
    // Constant value
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    // Identifier
    Ident(Rc<str>),
    // Pointer ref/deref
    Ref(Box<Expr>),
    Deref(Box<Expr>),
    // Unary operations
    Inv(Box<Expr>),
    Neg(Box<Expr>),
    // Postfix expressions
    Field(Box<Expr>, Rc<str>),
    Call(Box<Expr>, Vec<Expr>),
    // Binary operations
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Rem(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    Lsh(Box<Expr>, Box<Expr>),
    Rsh(Box<Expr>, Box<Expr>),
    // Cast
    Cast(Box<Expr>, Type)
}

impl Expr {
    // Evaluate expression as usize
    pub fn eval_usize(&self) -> usize {
        match self {
            Expr::U8(v)   => *v as usize,
            Expr::I8(v)   => *v as usize,
            Expr::U16(v)  => *v as usize,
            Expr::I16(v)  => *v as usize,
            Expr::U32(v)  => *v as usize,
            Expr::I32(v)  => *v as usize,
            Expr::U64(v)  => *v as usize,
            Expr::I64(v)  => *v as usize,
            _ => panic!("Can't evaluate non-constant expression at compile time"),
        }
    }
}

#[derive(Debug)]
pub enum Init {
    Base(Expr),
    List(Vec<Init>),
}

#[derive(Debug)]
pub enum Stmt {
    Eval(Expr),
    Ret(Expr),
    Auto(Rc<str>, Type, Option<Init>),
    Label(Rc<str>),
    Set(Expr, Expr),
    Jmp(Rc<str>),
    Jeq(Rc<str>, Expr, Expr),
    Jl(Rc<str>, Expr, Expr),
    Jle(Rc<str>, Expr, Expr),
    Jg(Rc<str>, Expr, Expr),
    Jge(Rc<str>, Expr, Expr),
}

#[derive(Debug,PartialEq)]
pub enum Vis {
    Private,    // Internal definition
    Export,     // Exported definition
    Extern,     // External definition reference
}

#[derive(Debug)]
pub struct Static {
    // Visibility
    pub vis: Vis,
    // Function name
    pub name: Rc<str>,
    // Type of static
    pub dtype: Type,
    // Initializer (if present)
    pub init: Option<Init>,
}

#[derive(Debug)]
pub struct Func {
    // Visibility
    pub vis: Vis,
    // Function name
    pub name: Rc<str>,
    // Parameters
    pub params: Vec<(Rc<str>, Type)>,
    pub varargs: bool,
    // Return type
    pub rettype: Type,
    // Statements
    pub stmts: Vec<Stmt>,
}

impl Func {
    pub fn new(vis: Vis, name: Rc<str>) -> Func {
        Func {
            vis: vis,
            name: name,
            params: Vec::new(),
            varargs: false,
            rettype: Type::VOID,
            stmts: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct File {
    pub records: HashMap<Rc<str>, Rc<Record>>,
    pub statics: HashMap<Rc<str>, Static>,
    pub funcs: HashMap<Rc<str>, Func>,
}

impl File {
    pub fn new() -> File {
        File {
            records: HashMap::new(),
            statics: HashMap::new(),
            funcs: HashMap::new(),
        }
    }
}
