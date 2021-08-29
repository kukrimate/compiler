use super::lex::{Lexer,Token};

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

// Size of a pointer
const PTR_SIZE: usize = 8;

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
}

//
// Functions
//

#[derive(Debug)]
pub enum Expr {
    // Constant value
    Const(IntVal),
    // Identifier (variable reference)
    Ident(Rc<str>),
    // Pointer ref/deref
    Ref(Box<Expr>),
    Deref(Box<Expr>),
    // Unary bitwise inversion
    Inv(Box<Expr>),
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
    pub locals: HashMap<Rc<str>, Type>,
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
    pub globals: HashMap<Rc<str>, (Type, bool)>,
    // Records/unions
    records: HashMap<Rc<str>, Rc<Record>>,
}

impl<'source> Parser<'source> {
    pub fn new(lex: &'source mut Lexer<'source>) -> Parser<'source> {
        Parser {
            tmp: lex.next(),
            lex: lex,
            globals: HashMap::new(),
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

        fn want_type(p: &mut Parser) -> Type {
            match std::mem::replace(&mut p.tmp, p.lex.next())? {
                Token::U8       => Type::U8,
                Token::I8       => Type::I8,
                Token::U16      => Type::U16,
                Token::I16      => Type::I16,
                Token::U32      => Type::U32,
                Token::I32      => Type::I32,
                Token::U64      => Type::U64,
                Token::I64      => Type::I64,
                Token::Mul      => {

                },
                Token::LSquare  => {

                },
                Token::Record   => {

                },
                Token::Union    => {

                },
                _ => panic!("Invalid typename!"),
            }
            Type::U8
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
                    let r#type = want_type(self);
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
