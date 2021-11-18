// SPDX-License-Identifier: GPL-2.0-only

//
// Recursive descent parser for the grammer described in "grammar.txt"
//

#![macro_use]

use crate::lex::{Lexer,Token};
use crate::gen::Gen;
use std::collections::HashMap;
use std::rc::Rc;

macro_rules! round_up {
    ($val:expr,$bound:expr) => {
        ($val + $bound - 1) / $bound * $bound
    }
}

//
// Type system
//

#[derive(Clone,Debug,PartialEq)]
pub enum Type {
    Void,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    Ptr {
        // What type does this point to?
        base_type: Box<Type>,
    },
    Array {
        // Type of array elements
        elem_type: Box<Type>,
        // Number of array elements
        elem_count: usize,
    },
    Record {
        // Was this declared as a union?
        is_union: bool,
        // Name lookup table
        lookup: HashMap<Rc<str>, usize>,
        // Field types and offsets (in declaration order)
        fields: Box<[(Type, usize)]>,
        // Pre-calculated alingment and size
        align: usize,
        size: usize,
    },
    Func {
        params: Box<[Type]>,
        varargs: bool,
        rettype: Box<Type>,
    },
}

impl Type {
    pub fn get_align(&self) -> usize {
        match self {
            Type::U8  => 1,
            Type::I8  => 1,
            Type::U16 => 2,
            Type::I16 => 2,
            Type::U32 => 4,
            Type::I32 => 4,
            Type::U64 => 8,
            Type::I64 => 8,
            Type::Ptr {..} => 8,
            Type::Array { elem_type, .. } => elem_type.get_align(),
            Type::Record { align, .. } => *align,
            Type::Void | Type::Func {..} => unreachable!(),
        }
    }

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
            Type::Ptr {..} => 8,
            Type::Array { elem_type, elem_count }
                => elem_type.get_size() * elem_count,
            Type::Record { size, .. } => *size,
            Type::Void | Type::Func {..} => unreachable!(),
        }
    }
}

//
// Abstract syntax tree elements
//

#[derive(Debug)]
pub enum Expr {
    // Constant value
    Const(Type, usize),
    // Reference to symbol
    Sym(Rc<str>),
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

    pub fn make_cast(self, new_type: Type) -> Expr {
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

    pub fn want_const(&self) -> usize {
        if let Expr::Const(_, val) = self {
            *val
        } else {
            panic!("Expected constant expression")
        }
    }
}

#[derive(Debug)]
pub enum Init {
    Base(Expr),
    List(Vec<Init>),
}

impl Init {
    pub fn want_expr(self) -> Expr {
        match self {
            Init::Base(expr) => expr,
            _ => panic!("Wanted bare initializer!"),
        }
    }

    pub fn want_list(self) -> Vec<Init> {
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
    Auto(Rc<str>, Type, Option<Init>),
    Label(Rc<str>),
    Set(Expr, Expr),
    Jmp(Rc<str>),
    Jeq(Rc<str>, Expr, Expr),
    Jneq(Rc<str>, Expr, Expr),
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

//
// Parser context
//

struct Parser<'source> {
    // Lexer and temorary token
    tmp: Option<Token>,
    lex: &'source mut Lexer<'source>,
    // Code generation backend
    gen: &'source mut Gen,
    // Currently defined record types
    records: HashMap<Rc<str>, Type>,
}

macro_rules! want {
    ($self:expr, $pattern:pat, $err:expr) => {
        match $self.tmp {
            Some($pattern) => $self.tmp = $self.lex.next(),
            _ => panic!("{}, got {:?}", $err, $self.tmp),
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
    fn new(lex: &'source mut Lexer<'source>, gen: &'source mut Gen) -> Parser<'source> {
        Parser {
            tmp: lex.next(),
            lex: lex,
            gen: gen,
            records: HashMap::new(),
        }
    }

    fn next_token(&mut self) -> Token {
        std::mem::replace(&mut self.tmp, self.lex.next()).unwrap()
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
            tok => panic!("Expected identifier, got {:?}!", tok),
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
            tok => panic!("Expected label, got {:?}!", tok),
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
            Token::U64  => Type::U64,
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
            Token::Str(data) => {
                let suffix = self.want_type_suffix();
                let string_sym = Expr::Sym(self.gen.do_string(suffix, &*data));
                // Replace string literal with reference to internal symbol
                Expr::Ref(Box::from(string_sym))
            },
            Token::Ident(s)
                => Expr::Sym(s),
            Token::Constant(val)
                => Expr::Const(self.want_type_suffix(), val),
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
                want!(self, Token::RSq, "Expected ]");
            } else {
                return expr;
            }
        }
    }

    fn want_unary(&mut self) -> Expr {
        if maybe_want!(self, Token::Sub) {
            self.want_unary().make_neg()
        } else if maybe_want!(self, Token::Tilde) {
            self.want_unary().make_inv()
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
        let expr = self.want_unary();
        if maybe_want!(self, Token::Cast) {
            expr.make_cast(self.want_type())
        } else {
            expr
        }
    }

    fn want_mul(&mut self) -> Expr {
        let expr = self.want_cast();
        if maybe_want!(self, Token::Mul) {
            expr.make_mul(self.want_mul())
        } else if maybe_want!(self, Token::Div) {
            expr.make_div(self.want_mul())
        } else if maybe_want!(self, Token::Rem) {
            expr.make_rem(self.want_mul())
        } else {
            expr
        }
    }

    fn want_add(&mut self) -> Expr {
        let expr = self.want_mul();
        if maybe_want!(self, Token::Add) {
            expr.make_add(self.want_add())
        } else if maybe_want!(self, Token::Sub) {
            expr.make_sub(self.want_add())
        } else {
            expr
        }
    }

    fn want_shift(&mut self) -> Expr {
        let expr = self.want_add();
        if maybe_want!(self, Token::Lsh) {
            expr.make_lsh(self.want_shift())
        } else if maybe_want!(self, Token::Rsh) {
            expr.make_rsh(self.want_shift())
        } else {
            expr
        }
    }

    fn want_and(&mut self) -> Expr {
        let expr = self.want_shift();
        if maybe_want!(self, Token::And) {
            expr.make_and(self.want_and())
        } else {
            expr
        }
    }

    fn want_xor(&mut self) -> Expr {
        let expr = self.want_and();
        if maybe_want!(self, Token::Xor) {
            expr.make_xor(self.want_xor())
        } else {
            expr
        }
    }

    fn want_or(&mut self) -> Expr {
        let expr = self.want_xor();
        if maybe_want!(self, Token::Or) {
            expr.make_or(self.want_or())
        } else {
            expr
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
            Token::Mul  => Type::Ptr {
                base_type: Box::new(self.want_type())
            },
            Token::LSq  => {
                let elem_type = self.want_type();
                want!(self, Token::Semicolon, "Expected ;");
                let elem_count_expr = self.want_expr();
                want!(self, Token::RSq, "Expected ]");
                Type::Array {
                    elem_type: Box::new(elem_type),
                    elem_count: match elem_count_expr {
                        Expr::Const(_, val) => val,
                        _ => panic!("Array element count must be constnat"),
                    }
                }
            },
            Token::Ident(ref ident) => {
                if let Some(record) = self.records.get(ident) {
                    return record.clone();
                } else {
                    panic!("Non-existent type {}", ident)
                }
            },
            Token::Fn => {
                // Read parameter types
                want!(self, Token::LParen, "Expected (");
                let mut params = Vec::new();
                let mut varargs = false;
                while !maybe_want!(self, Token::RParen) {
                    if maybe_want!(self, Token::Varargs) {
                        varargs = true;
                        want!(self, Token::LParen, "Expected )");
                        break;
                    }
                    params.push(self.want_type());
                    if !maybe_want!(self, Token::Comma) {
                        want!(self, Token::RParen, "Expected )");
                        break;
                    }
                }

                // Read return type
                let rettype = if maybe_want!(self, Token::Arrow) {
                    self.want_type()
                } else {
                    Type::Void
                };

                Type::Func {
                    params: params.into(),
                    varargs: varargs,
                    rettype: Box::new(rettype)
                }
            },
            _ => panic!("Invalid typename!"),
        }
    }

    fn want_record(&mut self, is_union: bool) -> Type {
        let mut lookup = HashMap::new();
        let mut fields = Vec::new();
        let mut max_align = 0;
        let mut offset = 0;

        want!(self, Token::LCurly, "Expected left curly");

        // Read fields until }
        while !maybe_want!(self, Token::RCurly) {
            lookup.insert(self.want_ident(), fields.len());
            want!(self, Token::Colon, "Expected :");

            let field_type = self.want_type();
            let field_align = field_type.get_align();
            let field_size = field_type.get_size();

            // Save maximum alignment of all fields
            if max_align < field_align {
                max_align = field_align;
            }

            // Round field offset to correct alignment
            offset = round_up!(offset, field_align);
            fields.push((field_type, offset));
            offset += field_size;

            if !maybe_want!(self, Token::Comma) {
                want!(self, Token::RCurly, "Expected right curly");
                break;
            }
        }

        // Record type declaration must end in semicolon
        want!(self, Token::Semicolon, "Expected ;");

        Type::Record {
            is_union: is_union,
            lookup: lookup,
            fields: fields.into_boxed_slice(),
            align: max_align,
            // Round struct size to a multiple of it's alignment, this is needed
            // if an array is ever created of this struct
            size: round_up!(offset, max_align),
        }
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
            Token::Ret      => {
                if maybe_want!(self, Token::Semicolon) {
                    // NOTE: return since we already have the semicolon
                    return Stmt::Ret(None)
                } else {
                    Stmt::Ret(Some(self.want_expr()))
                }
            },
            Token::Auto     => {
                let ident = self.want_ident();
                want!(self, Token::Colon, "Expected :");
                let dtype = self.want_type();
                let mut init = None;
                if maybe_want!(self, Token::Eq) {
                    init = Some(self.want_initializer());
                }
                Stmt::Auto(ident, dtype, init)
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
            Token::Jneq     => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jneq(label, expr1, self.want_expr())
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

    fn process(&mut self) {
        while !self.tmp.is_none() {
            match self.next_token() {
                Token::Record => {
                    let name = self.want_ident();
                    let record = self.want_record(false);
                    self.records.insert(name, record);
                },
                Token::Union => {
                    let name = self.want_ident();
                    let union = self.want_record(true);
                    self.records.insert(name, union);
                },
                Token::Static => {
                    let vis = self.maybe_want_vis();
                    let name = self.want_ident();
                    want!(self, Token::Colon, "Expected :");
                    let dtype = self.want_type();
                    if maybe_want!(self, Token::Eq) {
                        let init = self.want_initializer();
                        self.gen.do_static_init(vis, name, dtype, init);
                    } else {
                        self.gen.do_static(vis, name, dtype);
                    }
                    want!(self, Token::Semicolon, "Expected ;");
                },
                Token::Fn => {
                    let vis = self.maybe_want_vis();
                    let name = self.want_ident();

                    let mut params = Vec::new();
                    let mut varargs = false;

                    let mut param_tab = Vec::new();

                    // Read parameters
                    want!(self, Token::LParen, "Expected (");
                    while !maybe_want!(self, Token::RParen) {
                        // Last parameter can be varargs
                        if maybe_want!(self, Token::Varargs) {
                            varargs = true;
                            want!(self, Token::RParen, "Expected )");
                            break;
                        }
                        // Otherwise try reading a normal parameter
                        let param_name = self.want_ident();
                        want!(self, Token::Colon, "Expected :");
                        let param_type = self.want_type();
                        params.push(param_type.clone());
                        param_tab.push((param_name, param_type));
                        if !maybe_want!(self, Token::Comma) {
                            want!(self, Token::RParen, "Expected )");
                            break;
                        }
                    }

                    // Read return type (or set to void)
                    let rettype = if maybe_want!(self, Token::Arrow) {
                        self.want_type()
                    } else {
                        Type::Void
                    };

                    // Create symbol for function
                    self.gen.do_sym(vis, name.clone(), Type::Func {
                        params: params.into(),
                        varargs: varargs,
                        rettype: Box::new(rettype),
                    });

                    // Read body (if present)
                    let mut stmts = Vec::new();
                    if !maybe_want!(self, Token::Semicolon) {
                        want!(self, Token::LCurly, "Expected left curly");
                        while !maybe_want!(self, Token::RCurly) {
                            stmts.push(self.want_stmt());
                        }
                        // Generate body
                        self.gen.do_func(name, param_tab, stmts);
                    }
                },
                _ => panic!("Expected record, union, static or function!"),
            }
        }
    }
}

pub fn parse_file(data: &str, gen: &mut Gen) {
    let mut lexer = Lexer::new(data);
    let mut parser = Parser::new(&mut lexer, gen);
    parser.process();
}
