// SPDX-License-Identifier: GPL-2.0-only

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

pub const PTR_SIZE: usize = 8;

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
    // pub fn get_type(&self) -> Type {
    //     match self {
    //         IntVal::U8(_)   => Type::U8,
    //         IntVal::I8(_)   => Type::I8,
    //         IntVal::U16(_)  => Type::U16,
    //         IntVal::I16(_)  => Type::I16,
    //         IntVal::U32(_)  => Type::U32,
    //         IntVal::I32(_)  => Type::I32,
    //         IntVal::U64(_)  => Type::U64,
    //         IntVal::I64(_)  => Type::I64,
    //     }
    // }

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
// Functions
//

#[derive(Debug)]
pub enum Expr {
    // Constant value
    Const(IntVal),
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
    Cast(Box<Expr>, Type)
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
    pub r#type: Type,
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
    // Return type
    pub rettype: Option<Type>,
    // Statements
    pub stmts: Vec<Stmt>,
}

impl Func {
    fn new(vis: Vis, name: Rc<str>) -> Func {
        Func {
            vis: vis,
            name: name,
            params: Vec::new(),
            rettype: None,
            stmts: Vec::new(),
        }
    }
}

//
// Parser
//

pub struct Parser<'source> {
    // Lexer and temorary token
    tmp: Option<Token>,
    lex: &'source mut Lexer<'source>,
    // Records/unions
    records: HashMap<Rc<str>, Rc<Record>>,
    // Static variables
    statics: Vec<Static>,
    litno: usize,
}

macro_rules! want {
    ($self:expr, $pattern:pat, $err:expr) => {
        match $self.tmp {
            Some($pattern) => $self.tmp = $self.lex.next(),
            _ => panic!($err),
        }
    }
}

macro_rules! maybe_want {
    ($self:expr, $pattern:pat) => {
        match $self.tmp {
            Some($pattern) => {
                $self.tmp = $self.lex.next();
                true
            },
            _ => false,
        }
    }
}

impl<'source> Parser<'source> {
    pub fn new(lex: &'source mut Lexer<'source>) -> Parser<'source> {
        Parser {
            tmp: lex.next(),
            lex: lex,
            records: HashMap::new(),
            statics: Vec::new(),
            litno: 0,
        }
    }

    fn next_token(&mut self) -> Token {
        std::mem::replace(&mut self.tmp, self.lex.next()).unwrap()
    }

    // FIXME: take string literal type into account
    fn make_string_lit(&mut self, _: Type, data: Rc<str>) -> Rc<str> {
        // Create globally unique name for the literal
        let name: Rc<str> = Rc::from(format!("_slit_{}", self.litno));
        self.litno += 1;

        // Create NUL-terminated initializer for the string
        let mut list = Vec::new();
        for b in data.as_bytes() {
            list.push(Init::Base(Expr::Const(IntVal::U8(*b))));
        }
        list.push(Init::Base(Expr::Const(IntVal::U8(0))));

        // Create static variable for it
        self.statics.push(Static {
            vis: Vis::Private,
            name: name.clone(),
            r#type: Type::Array {
                elem_count: list.len(),
                elem_type: Box::from(Type::U8),
            },
            init: Some(Init::List(list)),
        });

        name
    }

    fn want_ident(&mut self) -> Rc<str> {
        match &self.tmp {
            Some(Token::Ident(_)) => {
                if let Token::Ident(s) = self.next_token() {
                    s
                } else {
                    unreachable!()
                }
            },
            tok @ _ => panic!("Expected identifier, got {:?}!", tok),
        }
    }

    fn want_label(&mut self) -> Rc<str> {
        match &self.tmp {
            Some(Token::Label(_)) => {
                if let Token::Label(s) = self.next_token() {
                    s
                } else {
                    unreachable!()
                }
            },
            tok @ _ => panic!("Expected label, got {:?}!", tok),
        }
    }

    fn maybe_want_vis(&mut self) -> Vis {
        if maybe_want!(self, Token::Export) {
            Vis::Export
        } else if maybe_want!(self, Token::Extern) {
            Vis::Extern
        } else {
            Vis::Private
        }
    }

    fn want_type_suffix(&mut self) -> Type {
        match self.next_token() {
            Token::U8   => Type::U8,
            Token::I8   => Type::I8,
            Token::U16  => Type::U16,
            Token::I16  => Type::I16,
            Token::U32  => Type::U32,
            Token::I32  => Type::I32,
            Token::U64  => Type::U32,
            Token::I64  => Type::I64,
            _ => panic!("Invalid type suffix!"),
        }
    }

    fn want_primary(&mut self) -> Expr {
        match self.next_token() {
            Token::LParen => {
                let expr = self.want_expr();
                want!(self, Token::RParen, "Missing )");
                expr
            },
            Token::Str(s) => {
                let r#type = self.want_type_suffix();
                Expr::Ident(self.make_string_lit(r#type, s))
            },
            Token::Ident(s) => Expr::Ident(s),
            Token::Constant(val) => match self.want_type_suffix() {
                Type::U8    => Expr::Const(IntVal::U8(val as u8)),
                Type::I8    => Expr::Const(IntVal::I8(val as i8)),
                Type::U16   => Expr::Const(IntVal::U16(val as u16)),
                Type::I16   => Expr::Const(IntVal::I16(val as i16)),
                Type::U32   => Expr::Const(IntVal::U32(val as u32)),
                Type::I32   => Expr::Const(IntVal::I32(val as i32)),
                Type::U64   => Expr::Const(IntVal::U64(val as u64)),
                Type::I64   => Expr::Const(IntVal::I64(val as i64)),
                _ => unreachable!(),
            },
            _ => panic!("Invalid constant value!"),
        }
    }

    fn want_postfix(&mut self) -> Expr {
        let mut expr = self.want_primary();
        loop {
            if maybe_want!(self, Token::Dot) {
                expr = Expr::Field(Box::from(expr), self.want_ident());
            } else if maybe_want!(self, Token::LParen) {
                let mut args = Vec::new();
                while !maybe_want!(self, Token::RParen) {
                    args.push(self.want_expr());
                    if !maybe_want!(self, Token::Comma) {
                        want!(self, Token::RParen, "Expected )");
                        break;
                    }
                }
                expr = Expr::Call(Box::from(expr), args);
            } else if maybe_want!(self, Token::LSq) {
                expr = Expr::Elem(Box::from(expr), Box::from(self.want_expr()));
            } else {
                return expr;
            }
        }
    }

    fn want_unary(&mut self) -> Expr {
        if maybe_want!(self, Token::Sub) {
            Expr::Neg(Box::from(self.want_unary()))
        } else if maybe_want!(self, Token::Tilde) {
            Expr::Inv(Box::from(self.want_unary()))
        } else if maybe_want!(self, Token::Mul) {
            Expr::Deref(Box::from(self.want_unary()))
        } else if maybe_want!(self, Token::And) {
            Expr::Ref(Box::from(self.want_unary()))
        } else if maybe_want!(self, Token::Add) {
            self.want_unary()
        } else {
            self.want_postfix()
        }
    }

    fn want_cast(&mut self) -> Expr {
        let expr1 = self.want_unary();
        if maybe_want!(self, Token::Cast) {
            Expr::Cast(Box::from(expr1), self.want_type())
        } else {
            expr1
        }
    }

    fn want_mul(&mut self) -> Expr {
        let expr1 = self.want_cast();
        if maybe_want!(self, Token::Mul) {
            Expr::Mul(Box::from(expr1), Box::from(self.want_mul()))
        } else if maybe_want!(self, Token::Div) {
            Expr::Div(Box::from(expr1), Box::from(self.want_mul()))
        } else if maybe_want!(self, Token::Rem) {
            Expr::Rem(Box::from(expr1), Box::from(self.want_mul()))
        } else {
            expr1
        }
    }

    fn want_add(&mut self) -> Expr {
        let expr1 = self.want_mul();
        if maybe_want!(self, Token::Add) {
            Expr::Add(Box::from(expr1), Box::from(self.want_add()))
        } else if maybe_want!(self, Token::Sub) {
            Expr::Sub(Box::from(expr1), Box::from(self.want_add()))
        } else {
            expr1
        }
    }

    fn want_shift(&mut self) -> Expr {
        let expr1 = self.want_add();
        if maybe_want!(self, Token::Lsh) {
            Expr::Lsh(Box::from(expr1), Box::from(self.want_shift()))
        } else if maybe_want!(self, Token::Rsh) {
            Expr::Rsh(Box::from(expr1), Box::from(self.want_shift()))
        } else {
            expr1
        }
    }

    fn want_and(&mut self) -> Expr {
        let expr1 = self.want_shift();
        if maybe_want!(self, Token::And) {
            Expr::And(Box::from(expr1), Box::from(self.want_and()))
        } else {
            expr1
        }
    }

    fn want_xor(&mut self) -> Expr {
        let expr1 = self.want_and();
        if maybe_want!(self, Token::Xor) {
            Expr::Xor(Box::from(expr1), Box::from(self.want_xor()))
        } else {
            expr1
        }
    }

    fn want_or(&mut self) -> Expr {
        let expr1 = self.want_xor();
        if maybe_want!(self, Token::Or) {
            Expr::Or(Box::from(expr1), Box::from(self.want_or()))
        } else {
            expr1
        }
    }

    fn want_expr(&mut self) -> Expr {
        self.want_or()
    }

    fn want_type(&mut self) -> Type {
        match self.next_token() {
            Token::U8   => Type::U8,
            Token::I8   => Type::I8,
            Token::U16  => Type::U16,
            Token::I16  => Type::I16,
            Token::U32  => Type::U32,
            Token::I32  => Type::I32,
            Token::U64  => Type::U64,
            Token::I64  => Type::I64,
            Token::Mul  => Type::Ptr { base_type: Box::new(self.want_type()) },
            Token::LSq  => {
                let elem_type = Box::new(self.want_type());
                want!(self, Token::Semicolon, "Expected ;");
                let elem_count_expr = self.want_expr();
                want!(self, Token::RSq, "Expected ]");
                let elem_count = match elem_count_expr {
                    Expr::Const(intval) => intval.as_usize(),
                    _ => panic!("Array size must be a constant!"),
                };
                Type::Array { elem_type: elem_type, elem_count: elem_count }
            },
            Token::Record   => {
                let ident = self.want_ident();
                if let Some(record) = self.records.get(&ident) {
                    if record.is_union {
                        panic!("Referencing union type with record keyword");
                    }
                    return Type::Record(record.clone());
                }
                panic!("Non-existent record type {}", ident)
            },
            Token::Union    => {
                let ident = self.want_ident();
                if let Some(union) = self.records.get(&ident) {
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

    fn want_record(&mut self, is_union: bool) -> Record {
        let mut r = Record {
            is_union: is_union,
            fields: HashMap::new()
        };
        // Read fields until }
        want!(self, Token::LCurly, "Expected left curly");
        while !maybe_want!(self, Token::RCurly) {
            let ident = self.want_ident();
            want!(self, Token::Colon, "Expected :");
            let r#type = self.want_type();
            r.fields.insert(ident, r#type);
            if !maybe_want!(self, Token::Comma) {
                want!(self, Token::RCurly, "Expected right curly");
                break;
            }
        }
        // Record type declaration must end in semicolon
        want!(self, Token::Semicolon, "Expected ;");
        r
    }

    fn want_initializer(&mut self) -> Init {
        if maybe_want!(self, Token::LCurly) {
            let mut list = Vec::new();
            while !maybe_want!(self, Token::RCurly) {
                list.push(self.want_initializer());
                if !maybe_want!(self, Token::Comma) {
                    want!(self, Token::RCurly, "Expected right curly");
                    break;
                }
            }
            Init::List(list)
        } else {
            Init::Base(self.want_expr())
        }
    }

    fn want_stmt(&mut self) -> Stmt {
        let stmt = match self.next_token() {
            Token::Eval     => Stmt::Eval(self.want_expr()),
            Token::Ret      => Stmt::Ret(self.want_expr()),
            Token::Auto     => {
                let ident = self.want_ident();
                want!(self, Token::Colon, "Expected :");
                let r#type = self.want_type();
                let mut init = None;
                if maybe_want!(self, Token::Eq) {
                    init = Some(self.want_initializer());
                }
                Stmt::Auto(ident, r#type, init)
            },
            Token::Label(s) => Stmt::Label(s),
            Token::Set      => {
                let var = self.want_expr();
                want!(self, Token::Eq, "Expected =");
                Stmt::Set(var, self.want_expr())
            },
            Token::Jmp      => Stmt::Jmp(self.want_label()),
            Token::Jeq      => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jeq(label, expr1, self.want_expr())
            },
            Token::Jl       => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jl(label, expr1, self.want_expr())
            },
            Token::Jle      => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jle(label, expr1, self.want_expr())
            },
            Token::Jg       => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jg(label, expr1, self.want_expr())
            },
            Token::Jge      => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jge(label, expr1, self.want_expr())
            },
            tok @ _ => panic!("Invalid statement: {:?}", tok),
        };
        if let Stmt::Label(_) = stmt {
            want!(self, Token::Colon, "Expected :");
        } else {
            want!(self, Token::Semicolon, "Expected ;");
        }
        stmt
    }

    pub fn parse_file(&mut self) -> (Vec<Static>, Vec<Func>) {
        let mut funcs = Vec::new();

        while !self.tmp.is_none() {
            match self.next_token() {
                Token::Record => {
                    let ident = self.want_ident();
                    let record = self.want_record(false);
                    self.records.insert(ident, Rc::from(record));
                },
                Token::Union => {
                    let ident = self.want_ident();
                    let union = self.want_record(true);
                    self.records.insert(ident, Rc::from(union));
                },
                Token::Static => {
                    let vis = self.maybe_want_vis();
                    let ident = self.want_ident();
                    want!(self, Token::Colon, "Expected :");
                    let r#type = self.want_type();
                    let mut init = None;
                    if maybe_want!(self, Token::Eq) {
                        init = Some(self.want_initializer());
                    }
                    want!(self, Token::Semicolon, "Expected ;");

                    self.statics.push(Static {
                        vis: vis,
                        name: ident,
                        r#type: r#type,
                        init: init
                    });
                },
                Token::Fn => {
                    let vis = self.maybe_want_vis();
                    let mut func = Func::new(vis, self.want_ident());

                    // Read parameters
                    want!(self, Token::LParen, "Expected (");
                    while !maybe_want!(self, Token::RParen) {
                        let ident = self.want_ident();
                        want!(self, Token::Colon, "Expected :");
                        let r#type = self.want_type();
                        func.params.push((ident, r#type));
                        if !maybe_want!(self, Token::Comma) {
                            want!(self, Token::RParen, "Expected )");
                            break;
                        }
                    }

                    // Read return type (if any)
                    if maybe_want!(self, Token::Arrow) {
                        func.rettype = Some(self.want_type());
                    }

                    // Read body (if present)
                    if !maybe_want!(self, Token::Semicolon) {
                        want!(self, Token::LCurly, "Expected left curly");
                        while !maybe_want!(self, Token::RCurly) {
                            func.stmts.push(self.want_stmt());
                        }
                    }

                    funcs.push(func);
                },
                _ => panic!("Expected record, union, static or function!"),
            }
        }

        (std::mem::replace(&mut self.statics, Vec::new()), funcs)
    }
}
