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

/*
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
*/

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
    Str(Type, Rc<str>),
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

#[derive(Debug)]
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
    pub params: HashMap<Rc<str>, Type>,
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
            params: HashMap::new(),
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
}

impl<'source> Parser<'source> {
    pub fn new(lex: &'source mut Lexer<'source>) -> Parser<'source> {
        Parser {
            tmp: lex.next(),
            lex: lex,
            records: HashMap::new(),
        }
    }

    fn next_token(&mut self) -> Token {
        std::mem::replace(&mut self.tmp, self.lex.next()).unwrap()
    }

    pub fn parse_file(&mut self) -> (Vec<Static>, Vec<Func>) {
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
            match &p.tmp {
                Some(Token::Ident(_)) => {
                    if let Token::Ident(s) = p.next_token() {
                        s
                    } else {
                        unreachable!()
                    }
                },
                tok @ _ => panic!("Expected identifier, got {:?}!", tok),
            }
        }

        fn want_label(p: &mut Parser) -> Rc<str> {
            match &p.tmp {
                Some(Token::Label(_)) => {
                    if let Token::Label(s) = p.next_token() {
                        s
                    } else {
                        unreachable!()
                    }
                },
                tok @ _ => panic!("Expected label, got {:?}!", tok),
            }
        }

        fn maybe_want_vis(p: &mut Parser) -> Vis {
            if maybe_want!(p, Token::Export) {
                Vis::Export
            } else if maybe_want!(p, Token::Extern) {
                Vis::Extern
            } else {
                Vis::Private
            }
        }

        fn want_type_suffix(p: &mut Parser) -> Type {
            match p.next_token() {
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

        fn want_primary(p: &mut Parser) -> Expr {
            match p.next_token() {
                Token::LParen => {
                    let expr = want_expr(p);
                    want!(p, Token::RParen, "Missing )");
                    expr
                },
                Token::Str(s) => Expr::Str(want_type_suffix(p), s),
                Token::Ident(s) => Expr::Ident(s),
                Token::Constant(val) => match want_type_suffix(p) {
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

        fn want_postfix(p: &mut Parser) -> Expr {
            let mut expr = want_primary(p);

            loop {
                if maybe_want!(p, Token::Dot) {
                    expr = Expr::Field(Box::from(expr), want_ident(p));
                } else if maybe_want!(p, Token::LParen) {
                    let mut args = Vec::new();
                    while !maybe_want!(p, Token::RParen) {
                        args.push(want_expr(p));
                        if !maybe_want!(p, Token::Comma) {
                            want!(p, Token::RParen, "Expected )");
                            break;
                        }
                    }
                    expr = Expr::Call(Box::from(expr), args);
                } else if maybe_want!(p, Token::LSq) {
                    expr = Expr::Elem(Box::from(expr), Box::from(want_expr(p)));
                } else {
                    return expr;
                }
            }
        }

        fn want_unary(p: &mut Parser) -> Expr {
            if maybe_want!(p, Token::Sub) {
                Expr::Neg(Box::from(want_unary(p)))
            } else if maybe_want!(p, Token::Tilde) {
                Expr::Inv(Box::from(want_unary(p)))
            } else if maybe_want!(p, Token::Mul) {
                Expr::Deref(Box::from(want_unary(p)))
            } else if maybe_want!(p, Token::And) {
                Expr::Ref(Box::from(want_unary(p)))
            } else if maybe_want!(p, Token::Add) {
                want_unary(p)
            } else {
                want_postfix(p)
            }
        }

        fn want_cast(p: &mut Parser) -> Expr {
            let expr1 = want_unary(p);
            if maybe_want!(p, Token::Cast) {
                Expr::Cast(Box::from(expr1), want_type(p))
            } else {
                expr1
            }
        }

        fn want_mul(p: &mut Parser) -> Expr {
            let expr1 = want_cast(p);
            if maybe_want!(p, Token::Mul) {
                Expr::Mul(Box::from(expr1), Box::from(want_mul(p)))
            } else if maybe_want!(p, Token::Div) {
                Expr::Div(Box::from(expr1), Box::from(want_mul(p)))
            } else if maybe_want!(p, Token::Rem) {
                Expr::Rem(Box::from(expr1), Box::from(want_mul(p)))
            } else {
                expr1
            }
        }

        fn want_add(p: &mut Parser) -> Expr {
            let expr1 = want_mul(p);
            if maybe_want!(p, Token::Add) {
                Expr::Add(Box::from(expr1), Box::from(want_add(p)))
            } else if maybe_want!(p, Token::Sub) {
                Expr::Sub(Box::from(expr1), Box::from(want_add(p)))
            } else {
                expr1
            }
        }

        fn want_shift(p: &mut Parser) -> Expr {
            let expr1 = want_add(p);
            if maybe_want!(p, Token::Lsh) {
                Expr::Lsh(Box::from(expr1), Box::from(want_shift(p)))
            } else if maybe_want!(p, Token::Rsh) {
                Expr::Rsh(Box::from(expr1), Box::from(want_shift(p)))
            } else {
                expr1
            }
        }

        fn want_and(p: &mut Parser) -> Expr {
            let expr1 = want_shift(p);
            if maybe_want!(p, Token::And) {
                Expr::And(Box::from(expr1), Box::from(want_and(p)))
            } else {
                expr1
            }
        }

        fn want_xor(p: &mut Parser) -> Expr {
            let expr1 = want_and(p);
            if maybe_want!(p, Token::Xor) {
                Expr::Xor(Box::from(expr1), Box::from(want_xor(p)))
            } else {
                expr1
            }
        }

        fn want_or(p: &mut Parser) -> Expr {
            let expr1 = want_xor(p);
            if maybe_want!(p, Token::Or) {
                Expr::Or(Box::from(expr1), Box::from(want_or(p)))
            } else {
                expr1
            }
        }

        fn want_expr(p: &mut Parser) -> Expr {
            want_or(p)
        }

        fn want_type(p: &mut Parser) -> Type {
            match p.next_token() {
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
                    let elem_count_expr = want_expr(p);
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
            want!(p, Token::LCurly, "Expected left curly");
            while !maybe_want!(p, Token::RCurly) {
                let ident = want_ident(p);
                want!(p, Token::Colon, "Expected :");
                let r#type = want_type(p);
                r.fields.insert(ident, r#type);
                if !maybe_want!(p, Token::Comma) {
                    want!(p, Token::RCurly, "Expected right curly");
                    break;
                }
            }
            // Record type declaration must end in semicolon
            want!(p, Token::Semicolon, "Expected ;");
            r
        }

        fn want_initializer(p: &mut Parser) -> Init {
            if maybe_want!(p, Token::LCurly) {
                let mut list = Vec::new();
                while !maybe_want!(p, Token::RCurly) {
                    list.push(want_initializer(p));
                    if !maybe_want!(p, Token::Comma) {
                        want!(p, Token::RCurly, "Expected right curly");
                        break;
                    }
                }
                Init::List(list)
            } else {
                Init::Base(want_expr(p))
            }
        }

        fn want_stmt(p: &mut Parser) -> Stmt {
            let stmt = match std::mem::replace(
                    &mut p.tmp, p.lex.next()).unwrap() {
                Token::Eval     => Stmt::Eval(want_expr(p)),
                Token::Ret      => Stmt::Ret(want_expr(p)),
                Token::Auto     => {
                    let ident = want_ident(p);
                    want!(p, Token::Colon, "Expected :");
                    let r#type = want_type(p);
                    let mut init = None;
                    if maybe_want!(p, Token::Eq) {
                        init = Some(want_initializer(p));
                    }
                    Stmt::Auto(ident, r#type, init)
                },
                Token::Label(s) => Stmt::Label(s),
                Token::Set      => {
                    let var = want_expr(p);
                    want!(p, Token::Eq, "Expected =");
                    Stmt::Set(var, want_expr(p))
                },
                Token::Jmp      => Stmt::Jmp(want_label(p)),
                Token::Jeq      => {
                    let label = want_label(p);
                    want!(p, Token::Comma, "Expected ,");
                    let expr1 = want_expr(p);
                    want!(p, Token::Comma, "Expected ,");
                    Stmt::Jeq(label, expr1, want_expr(p))
                },
                Token::Jl       => {
                    let label = want_label(p);
                    want!(p, Token::Comma, "Expected ,");
                    let expr1 = want_expr(p);
                    want!(p, Token::Comma, "Expected ,");
                    Stmt::Jl(label, expr1, want_expr(p))
                },
                Token::Jle      => {
                    let label = want_label(p);
                    want!(p, Token::Comma, "Expected ,");
                    let expr1 = want_expr(p);
                    want!(p, Token::Comma, "Expected ,");
                    Stmt::Jle(label, expr1, want_expr(p))
                },
                Token::Jg       => {
                    let label = want_label(p);
                    want!(p, Token::Comma, "Expected ,");
                    let expr1 = want_expr(p);
                    want!(p, Token::Comma, "Expected ,");
                    Stmt::Jg(label, expr1, want_expr(p))
                },
                Token::Jge      => {
                    let label = want_label(p);
                    want!(p, Token::Comma, "Expected ,");
                    let expr1 = want_expr(p);
                    want!(p, Token::Comma, "Expected ,");
                    Stmt::Jge(label, expr1, want_expr(p))
                },
                tok @ _ => panic!("Invalid statement: {:?}", tok),
            };
            if let Stmt::Label(_) = stmt {
                want!(p, Token::Colon, "Expected :");
            } else {
                want!(p, Token::Semicolon, "Expected ;");
            }
            stmt
        }

        let mut statics = Vec::new();
        let mut funcs = Vec::new();

        while !self.tmp.is_none() {
            match self.next_token() {
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
                    let vis = maybe_want_vis(self);
                    let ident = want_ident(self);
                    want!(self, Token::Colon, "Expected :");
                    let r#type = want_type(self);
                    let mut init = None;
                    if maybe_want!(self, Token::Eq) {
                        init = Some(want_initializer(self));
                    }
                    want!(self, Token::Semicolon, "Expected ;");

                    statics.push(Static {
                        vis: vis,
                        name: ident,
                        r#type: r#type,
                        init: init
                    });
                },
                Token::Fn => {
                    let vis = maybe_want_vis(self);
                    let mut func = Func::new(vis, want_ident(self));

                    // Read parameters
                    want!(self, Token::LParen, "Expected (");
                    while !maybe_want!(self, Token::RParen) {
                        let ident = want_ident(self);
                        want!(self, Token::Colon, "Expected :");
                        let r#type = want_type(self);
                        func.params.insert(ident, r#type);
                        if !maybe_want!(self, Token::Comma) {
                            want!(self, Token::RParen, "Expected )");
                            break;
                        }
                    }

                    // Read return type (if any)
                    if maybe_want!(self, Token::Arrow) {
                        func.rettype = Some(want_type(self));
                    }

                    // Read body (if present)
                    if !maybe_want!(self, Token::Semicolon) {
                        want!(self, Token::LCurly, "Expected left curly");
                        while !maybe_want!(self, Token::RCurly) {
                            func.stmts.push(want_stmt(self));
                        }
                    }

                    funcs.push(func);
                },
                _ => panic!("Expected record, union, static or function!"),
            }
        }

        (statics, funcs)
    }
}
