use super::lex::{Lexer,Token};
use super::gen::{Gen,PTR_SIZE,Storage};

use std::collections::HashMap;
use std::rc::Rc;

//
// Type system
//

#[derive(Debug)]
pub struct Record {
    is_union: bool,
    fields: HashMap<Rc<str>, Type>,
}

#[derive(Debug)]
pub enum Type {
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

impl Type {
    pub fn get_size(&self) -> usize {
        match self {
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
            Type::Record(record) => {
                let size_iter = record.fields.iter().map(
                    |(_, t)| -> usize {
                    t.get_size()
                });
                if record.is_union {
                    size_iter.max().unwrap_or_else(|| -> usize { 0 })
                } else {
                    size_iter.sum()
                }
            }
        }
    }
}

//
// Integer constant value
//

#[derive(Debug)]
pub enum IntVal {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
}

impl IntVal {
    pub fn get_type(&self) -> Type {
        match self {
            IntVal::U8(_)   => Type::U8,
            IntVal::I8(_)   => Type::I8,
            IntVal::U16(_)  => Type::U16,
            IntVal::I16(_)  => Type::I16,
            IntVal::U32(_)  => Type::U32,
            IntVal::I32(_)  => Type::I32,
            IntVal::U64(_)  => Type::U64,
            IntVal::I64(_)  => Type::I64,
        }
    }

    pub fn as_usize(&self) -> usize {
        match self {
            IntVal::U8(v)   => *v as usize,
            IntVal::I8(v)   => *v as usize,
            IntVal::U16(v)  => *v as usize,
            IntVal::I16(v)  => *v as usize,
            IntVal::U32(v)  => *v as usize,
            IntVal::I32(v)  => *v as usize,
            IntVal::U64(v)  => *v as usize,
            IntVal::I64(v)  => *v as usize,
        }
    }
}

//
// Variables
//

type Auto = (Type, usize);
type Static = (Type, Storage);

//
// Functions
//

#[derive(Debug)]
pub enum Var {
    Auto(Rc<Auto>),
    Static(Rc<Static>),
}

#[derive(Debug)]
pub enum Expr {
    // Constant value
    Const(IntVal),
    // Variable reference
    Var(Var),
    // Pointer ref/deref
    Ref(Box<Expr>),
    Deref(Box<Expr>),
    // Unary operations
    Inv(Box<Expr>),
    Neg(Box<Expr>),
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
    // Field of a record/union
    Field(Box<Expr>, Rc<str>),
    // Cast
    Cast(Box<Expr>, Type)
}

#[derive(Debug)]
pub enum Stmt {
    Label(Rc<str>),
    Set(Expr, Expr),
    Jmp(Rc<str>),
    Jeq(Rc<str>, Expr, Expr),
    Jl(Rc<str>, Expr, Expr),
    Jle(Rc<str>, Expr, Expr),
    Jg(Rc<str>, Expr, Expr),
    Jge(Rc<str>, Expr, Expr),
}

#[derive(Debug)]
pub struct Func {
    // Function name
    pub name: Rc<str>,
    // Local variables
    pub locals: HashMap<Rc<str>, Var>,
    // Statements
    pub stmts: Vec<Stmt>,
}

//
// Parser
//

pub struct Parser<'source> {
    tmp: Option<Token>,
    lex: &'source mut Lexer<'source>,
    // Global variables
    pub statics: HashMap<Rc<str>, Rc<Static>>,
    // Records/unions
    records: HashMap<Rc<str>, Rc<Record>>,
}

impl<'source> Parser<'source> {
    pub fn new(lex: &'source mut Lexer<'source>, gen: &'source mut Gen) -> Parser<'source> {
        Parser {
            tmp: lex.next(),
            lex: lex,
            statics: HashMap::new(),
            records: HashMap::new(),
        }
    }
}

impl<'source> Iterator for Parser<'source> {
    type Item = Func;

    fn next(&mut self) -> Option<Self::Item> {
        macro_rules! want {
            ($p:expr, $pattern:pat, $err:expr) => {
                match $p.tmp {
                    Some($pattern) => $p.tmp = $p.lex.next(),
                    _ => panic!($err),
                }
            }
        }

        macro_rules! maybe_want {
            ($p:expr, $pattern:pat) => {
                match $p.tmp {
                    Some($pattern) => {
                        $p.tmp = $p.lex.next();
                        true
                    },
                    _ => false,
                }
            }
        }

        fn want_ident(p: &mut Parser) -> Rc<str> {
            match p.tmp {
                Some(Token::Ident(_)) => {
                    if let Some(Token::Ident(s))
                            = std::mem::replace(&mut p.tmp, p.lex.next()) {
                        s
                    } else {
                        panic!("UNREACHABLE")
                    }
                },
                _ => panic!("Expected identifier!"),
            }
        }

        fn want_constant(p: &mut Parser) -> Expr {
            let val = match std::mem::replace(&mut p.tmp, p.lex.next()).unwrap() {
                Token::Constant(val) => val,
                _ => panic!("Invalid constant value!"),
            };
            match std::mem::replace(&mut p.tmp, p.lex.next()).unwrap() {
                Token::U8   => Expr::Const(IntVal::U8(val as u8)),
                Token::I8   => Expr::Const(IntVal::I8(val as i8)),
                Token::U16  => Expr::Const(IntVal::U16(val as u16)),
                Token::I16  => Expr::Const(IntVal::I16(val as i16)),
                Token::U32  => Expr::Const(IntVal::U32(val as u32)),
                Token::I32  => Expr::Const(IntVal::I32(val as i32)),
                Token::U64  => Expr::Const(IntVal::U64(val as u64)),
                Token::I64  => Expr::Const(IntVal::I64(val as i64)),
                _ => panic!("Invalid constant suffix!"),
            }
        }

        fn want_expr(p: &mut Parser) -> Expr {

        }

        fn want_type(p: &mut Parser) -> Type {
            match std::mem::replace(&mut p.tmp, p.lex.next()).unwrap() {
                Token::U8   => Type::U8,
                Token::I8   => Type::I8,
                Token::U16  => Type::U16,
                Token::I16  => Type::I16,
                Token::U32  => Type::U32,
                Token::I32  => Type::I32,
                Token::U64  => Type::U64,
                Token::I64  => Type::I64,
                Token::Mul  => Type::Ptr { base_type: Box::new(want_type(p)) },
                Token::LSq  => {
                    let elem_type = Box::new(want_type(p));
                    want!(p, Token::Semicolon, "Expected ;");
                    let elem_count_expr = want_constant(p);
                    want!(p, Token::RSq, "Expected ]");
                    let elem_count = match elem_count_expr {
                        Expr::Const(intval) => intval.as_usize(),
                        _ => panic!("Array size must be a constant!"),
                    };
                    Type::Array { elem_type: elem_type, elem_count: elem_count }
                },
                Token::Record   => {
                    let ident = want_ident(p);
                    if let Some(record) = p.records.get(&ident) {
                        if record.is_union {
                            panic!("Referencing union type with record keyword");
                        }
                        return Type::Record(record.clone());
                    }
                    panic!("Non-existent record type {}", ident)
                },
                Token::Union    => {
                    let ident = want_ident(p);
                    if let Some(union) = p.records.get(&ident) {
                        if !union.is_union {
                            panic!("Referencing record type with union keyword");
                        }
                        return Type::Record(union.clone());
                    }
                    panic!("Non-existent union type {}", ident)
                },
                _ => panic!("Invalid typename!"),
            }
        }

        fn want_record(p: &mut Parser, is_union: bool) -> Record {
            let mut r = Record {
                is_union: is_union,
                fields: HashMap::new()
            };
            // Read fields until }
            while !maybe_want!(p, Token::RSq) {
                let ident = want_ident(p);
                want!(p, Token::Colon, "Expected :");
                let r#type = want_type(p);
                r.fields.insert(ident, r#type);
                if !maybe_want!(p, Token::Comma) {
                    want!(p, Token::RSq, "Expected right curly");
                    break;
                }
            }
            // Record type declaration must end in semicolon
            want!(p, Token::Semicolon, "Expected ;");
            r
        }

        loop {
            match std::mem::replace(&mut self.tmp, self.lex.next())? {
                Token::Record => {
                    let ident = want_ident(self);
                    let record = want_record(self, false);
                    self.records.insert(ident, Rc::from(record));
                },
                Token::Union => {
                    let ident = want_ident(self);
                    let union = want_record(self, true);
                    self.records.insert(ident, Rc::from(union));
                },
                Token::Static => {
                    let export = maybe_want!(self, Token::Export);
                    let ident = want_ident(self);
                    want!(self, Token::Colon, "Expected :");
                    let r#type = want_type(self);
                    println!("static {:?}: {:?}", ident, r#type);
                    if maybe_want!(self, Token::Eq) {
                        panic!("FIXME: initializer list!");
                    }
                    want!(self, Token::Semicolon, "Expected ;")
                },
                Token::Fn => {
                    let export = maybe_want!(self, Token::Export);
                    let ident = want_ident(self);
                },
                _ => panic!("Expected record, union, static or function!"),
            }
        }
    }
}
