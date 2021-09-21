// SPDX-License-Identifier: GPL-2.0-only

//
// Abstract syntax tree elements
//

use std::collections::HashMap;
use std::rc::Rc;

//
// Type system
//

#[derive(Debug,PartialEq)]
pub struct Record {
    pub fields: HashMap<Rc<str>, (Rc<Type>, usize)>,
    pub size: usize,
}

#[derive(Clone,Debug,PartialEq)]
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
        base_type: Rc<Type>,
    },
    Array {
        elem_type: Rc<Type>,
        elem_count: usize,
    },
    Record(Rc<Record>),
}

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
            Type::Ptr {..} => 8,
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
    Const(Rc<Type>, usize),
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
    Elem(Box<Expr>, Box<Expr>),
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
    Cast(Box<Expr>, Rc<Type>)
}

//
// Constructors for expression that might be able to fold
//

impl Expr {
    pub fn make_inv(self) -> Expr {
        match self {
            Expr::Const(dtype, val) => Expr::Const(dtype, !val),
            expr => Expr::Inv(Box::from(expr)),
        }
    }

    pub fn make_neg(self) -> Expr {
        match self {
            Expr::Const(dtype, val) => Expr::Const(dtype, !val + 1),
            expr => Expr::Neg(Box::from(expr)),
        }
    }

    pub fn make_cast(self, new_type: Rc<Type>) -> Expr {
        match self {
            Expr::Const(_, val) => Expr::Const(new_type, val),
            expr => Expr::Cast(Box::from(expr), new_type)
        }
    }

    pub fn make_mul(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot multiply constants with different types")
                }
                Expr::Const(dtype1, val1 * val2)
            },
            expr1 => Expr::Mul(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_div(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot divide constants with different types")
                }
                Expr::Const(dtype1, val1 / val2)
            },
            expr1 => Expr::Div(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_rem(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot calculate remainder of constants with different types")
                }
                Expr::Const(dtype1, val1 % val2)
            },
            expr1 => Expr::Rem(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_add(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot add constants with different types")
                }
                Expr::Const(dtype1, val1 + val2)
            },
            expr1 => Expr::Add(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_sub(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot substract constants with different types")
                }
                Expr::Const(dtype1, val1 - val2)
            },
            expr1 => Expr::Sub(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_lsh(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                Expr::Const(dtype1, val1 << val2)
            },
            expr1 => Expr::Lsh(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_rsh(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                Expr::Const(dtype1, val1 >> val2)
            },
            expr1 => Expr::Rsh(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_and(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot bitwise-and constants with different types")
                }
                Expr::Const(dtype1, val1 & val2)
            },
            expr1 => Expr::And(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_xor(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot bitwise-xor constants with different types")
                }
                Expr::Const(dtype1, val1 ^ val2)
            },
            expr1 => Expr::Xor(Box::from(expr1), Box::from(expr2)),
        }
    }

    pub fn make_or(self, expr2: Expr) -> Expr {
        match self {
            Expr::Const(dtype1, val1) if let Expr::Const(dtype2, val2) = expr2 => {
                if dtype1 != dtype2 {
                    panic!("Cannot bitwise-xor constants with different types")
                }
                Expr::Const(dtype1, val1 | val2)
            },
            expr1 => Expr::Or(Box::from(expr1), Box::from(expr2)),
        }
    }
}

#[derive(Debug)]
pub enum Init {
    Base(Expr),
    List(Vec<Init>),
}

impl Init {
    pub fn want_expr(&self) -> &Expr {
        match self {
            Init::Base(expr) => expr,
            _ => panic!("Wanted bare initializer!"),
        }
    }

    pub fn want_list(&self) -> &Vec<Init> {
        match self {
            Init::List(list) => list,
            _ => panic!("Wanted initializer list!"),
        }
    }
}

#[derive(Debug)]
pub enum Stmt {
    Eval(Expr),
    Ret(Option<Expr>),
    Auto(Rc<str>, Rc<Type>, Option<Init>),
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
    pub params: Vec<(Rc<str>, Rc<Type>)>,
    pub varargs: bool,
    // Return type
    pub rettype: Rc<Type>,
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
            rettype: Rc::from(Type::VOID),
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
