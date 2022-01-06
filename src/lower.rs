// SPDX-License-Identifier: GPL-2.0-only

//
// AST to code generator "bridge"
//

use crate::ast::{Ty,Local,ExprKind,Expr,Stmt};
use crate::gen::{DataGen,UOp,BOp,Cond,FuncGen};
use std::collections::HashMap;
use std::rc::Rc;

//
// Lower an constant expression into data generation calls
//

pub struct DataLower<T: DataGen> {
	// Code generation backend
    gen: T,
}

impl<T: DataGen> DataLower<T> {
    pub fn new(gen: T) -> Self {
        DataLower {
            gen: gen,
        }
    }

    pub fn end(self) -> T {
        self.gen
    }

	pub fn expr(&mut self, expr: &Expr) {
		match &expr.kind {
            ExprKind::Const(val) => {
            	self.gen._const(&expr.ty, *val);
            },
            ExprKind::Compound(exprs) => {
                for expr in exprs.iter() {
                    self.expr(expr);
                }
            },
            ExprKind::Ref(expr) => {
                if let ExprKind::Global(name) = &expr.kind {
                	self.gen.addr(name);
                } else {
                    panic!("Expected constant expression")
                }
            },
            _ => panic!("Expected constant expression"),
        }
	}
}

//
// Lower a high level function into code generation calls
//

pub struct FuncLower<T: FuncGen> {
    // Code generation backend
    gen: T,
    // Local variable to storage mapping
    local_vals: HashMap<Local, T::Val>,
    // User label to unique label mapping
    user_labels: HashMap<Rc<str>, T::Label>,
    // Label for returning
    return_label: T::Label,
}

impl<T: FuncGen> FuncLower<T> {
    pub fn new(mut gen: T) -> Self {
        let return_label = gen.next_label();
        FuncLower {
            gen: gen,
            local_vals: HashMap::new(),
            user_labels: HashMap::new(),
            return_label: return_label,
        }
    }

    pub fn end(mut self) -> T {
        self.gen.tgt(self.return_label);
        self.gen
    }

    fn expr(&mut self, expr: &Expr) -> T::Val {
        match &expr.kind {
            // Constant value
            ExprKind::Const(val) => self.gen.immed(*val),
            // Compound literals
            ExprKind::Compound(exprs) => {
                let vals = exprs.iter()
                                .map(|expr| (&expr.ty, self.expr(expr)))
                                .collect();
                self.gen.compound(&expr.ty, vals)
            },
            // Reference to symbol
            ExprKind::Global(name)
                => self.gen.global(name.clone()),
            ExprKind::Local(local)
                => self.local_vals.get(local).cloned().unwrap(),

            // Postfix expressions
            ExprKind::Field(inner, off) => {
                let val = self.expr(&*inner);
                self.gen.field(val, *off)
            },

            ExprKind::Elem(array, index) => {
                let array = self.expr(&*array);
                let index = self.expr(&*index);
                self.gen.elem(array, index, expr.ty.get_size())
            },
            ExprKind::Call(func, args) => {
                let varargs = if let Ty::Func { varargs, .. } = &func.ty {
                    *varargs
                } else {
                    unreachable!();
                };
                let func = self.expr(&*func);
                let args = args.iter()
                               .map(|arg| (&arg.ty, self.expr(arg)))
                               .collect();
                self.gen.call(&expr.ty, varargs, func, args)
            },

            // Prefix expressions
            ExprKind::Ref(inner) => {
                let val = self.expr(&*inner);
                self.gen._ref(val)
            },
            ExprKind::Deref(inner) => {
                let val = self.expr(&*inner);
                self.gen.deref(val)
            },
            ExprKind::Not(inner) => {
                let val = self.expr(&*inner);
                self.gen.unary(UOp::Not, &inner.ty, val)
            },
            ExprKind::Neg(inner) => {
                let val = self.expr(&*inner);
                self.gen.unary(UOp::Neg, &inner.ty, val)
            },

            // Binary operations
            ExprKind::Mul(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Mul, &lhs.ty, v1, v2)
            },
            ExprKind::Div(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Div, &lhs.ty, v1, v2)
            },
            ExprKind::Rem(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Rem, &lhs.ty, v1, v2)
            },
            ExprKind::Add(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Add, &lhs.ty, v1, v2)
            },
            ExprKind::Sub(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Sub, &lhs.ty, v1, v2)
            },
            ExprKind::Lsh(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Lsh, &lhs.ty, v1, v2)
            },
            ExprKind::Rsh(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Rsh, &lhs.ty, v1, v2)
            },
            ExprKind::And(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::And, &lhs.ty, v1, v2)
            },
            ExprKind::Xor(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Xor, &lhs.ty, v1, v2)
            },
            ExprKind::Or(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.binary(BOp::Or, &lhs.ty, v1, v2)
            },

            // Boolean expressions
            ExprKind::LNot(_) |
            ExprKind::Lt(_, _) |
            ExprKind::Le(_, _) |
            ExprKind::Gt(_, _) |
            ExprKind::Ge(_, _) |
            ExprKind::Eq(_, _) |
            ExprKind::Ne(_, _) |
            ExprKind::LAnd(_, _) |
            ExprKind::LOr(_, _) => {
                let ltrue = self.gen.next_label();
                let lfalse = self.gen.next_label();
                self.bool_expr(expr, ltrue, lfalse);
                self.gen.bool_to_val(ltrue, lfalse)
            },

            // Cast
            ExprKind::Cast(inner) => {
                // FIXME: integer casts cannot be done this way
                self.expr(&*inner)
            },
        }
    }

    fn bool_expr(&mut self, expr: &Expr, ltrue: T::Label, lfalse: T::Label) {
        match &expr.kind {
            ExprKind::LNot(inner) => {
                self.bool_expr(&*inner, lfalse, ltrue)
            },
            ExprKind::Lt(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.jcc(Cond::Lt, ltrue, lfalse, &lhs.ty, v1, v2);
            },
            ExprKind::Le(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.jcc(Cond::Le, ltrue, lfalse, &lhs.ty, v1, v2);
            },
            ExprKind::Gt(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.jcc(Cond::Gt, ltrue, lfalse, &lhs.ty, v1, v2);
            },
            ExprKind::Ge(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.jcc(Cond::Ge, ltrue, lfalse, &lhs.ty, v1, v2);
            },
            ExprKind::Eq(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.jcc(Cond::Eq, ltrue, lfalse, &lhs.ty, v1, v2);
            },
            ExprKind::Ne(lhs, rhs) => {
                let v1 = self.expr(&*lhs);
                let v2 = self.expr(&*rhs);
                self.gen.jcc(Cond::Ne, ltrue, lfalse, &lhs.ty, v1, v2);
            },
            ExprKind::LAnd(lhs, rhs) => {
                let lmid = self.gen.next_label();
                self.bool_expr(&*lhs, lmid, lfalse);
                self.gen.tgt(lmid);
                self.bool_expr(&*rhs, ltrue, lfalse);
            },
            ExprKind::LOr(lhs, rhs) => {
                let lmid = self.gen.next_label();
                self.bool_expr(&*lhs, ltrue, lmid);
                self.gen.tgt(lmid);
                self.bool_expr(&*rhs, ltrue, lfalse);
            }
            _ => {
                let val = self.expr(expr);
                self.gen.val_to_bool(val, ltrue, lfalse);
            },
        }
    }

    pub fn param(&mut self, local: Local, ty: &Ty) {
        let val = self.gen.next_local(ty);
        // Move paramater to local variable
        self.gen.arg_to_val(ty, val.clone());
        // Add local variable storage to table
        self.local_vals.insert(local, val);
    }

    pub fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    self.stmt(stmt);
                }
            },

            Stmt::Eval(expr) => {
                self.expr(expr);
            },

            Stmt::Ret(opt_expr) => {
                // Evaluate return value if present
                if let Some(expr) = opt_expr {
                    let val = self.expr(expr);
                    self.gen.val_to_ret(&expr.ty, val);
                }
                // Jump to the end of function
                self.gen.jmp(self.return_label);
            },

            Stmt::Auto(ty, local, opt_expr) => {
                let val = self.gen.next_local(ty);
                // Make not of where the local variable is stored
                self.local_vals.insert(*local, val.clone());
                // Initialze variable if required
                if let Some(expr) = opt_expr {
                    let src = self.expr(expr);
                    self.gen.assign(&expr.ty, val, src);
                }
            },

            Stmt::Label(name) => {
                if let Some(_) = self.user_labels.get(name) {
                    panic!("Duplicate label {}", name);
                }
                // Create a mapping from the name to the label
                let label = self.gen.next_label();
                self.user_labels.insert(name.clone(), label);
                // Generate jump target
                self.gen.tgt(label);
            },

            Stmt::Set(dst, src) => {
                // Find source and destination value
                let dval = self.expr(dst);
                let sval = self.expr(src);
                // Perform copy
                self.gen.assign(&dst.ty, dval, sval);
            },

            Stmt::Jmp(name) => {
                if let Some(label) = self.user_labels.get(name) {
                    // Generate jump to label
                    self.gen.jmp(*label);
                } else {
                    panic!("Unknown label {}", name);
                }
            },

            Stmt::If(cond, then, opt_else) => {
                let lthen = self.gen.next_label();
                let lelse = self.gen.next_label();
                let lend = self.gen.next_label();

                // Jump to correct case based on condition
                self.bool_expr(cond, lthen, lelse);

                // Generate then case
                self.gen.tgt(lthen);
                self.stmt(then);

                // Jump to end after true case
                self.gen.jmp(lend);

                // Generate else case
                self.gen.tgt(lelse);
                if let Some(_else) = opt_else {
                    self.stmt(_else);
                }

                // Jump target for end
                self.gen.tgt(lend);
            },

            Stmt::While(cond, body) => {
                let ltest = self.gen.next_label();
                let lbody = self.gen.next_label();
                let lend = self.gen.next_label();
                // Do the test (it either jumps to the body or the end)
                self.gen.tgt(ltest);
                self.bool_expr(cond, lbody, lend);
                // Generate body
                self.gen.tgt(lbody);
                self.stmt(body);
                // Test again after the body
                self.gen.jmp(ltest);
                // False test jumps here
                self.gen.tgt(lend);
            },
        }
    }
}
