// SPDX-License-Identifier: GPL-2.0-only

//
// Symbol table
//

use super::Expr;
use std::collections::HashMap;
use std::rc::Rc;

//
// Symbol visibility
//

#[derive(Debug,PartialEq)]
pub enum Vis {
    Private,    // Internal definition
    Export,     // Exported definition
    Extern,     // External definition reference
}

//
// Chained hash tables used for a symbol table
//

pub struct SymTab {
    list: Vec<HashMap<Rc<str>, Expr>>,
}

impl SymTab {
    pub fn new() -> SymTab {
        let mut symtab = SymTab {
            list: Vec::new(),
        };
        symtab.list.push(HashMap::new());
        symtab
    }

    pub fn insert(&mut self, name: Rc<str>, expr: Expr) {
        let scope = self.list.last_mut().unwrap();
        if let None = scope.get(&name) {
            scope.insert(name, expr);
        } else {
            panic!("Re-declaration of {}", name)
        }
    }

    pub fn lookup(&mut self, name: &Rc<str>) -> &Expr {
        for scope in self.list.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return ty;
            }
        }
        panic!("Unknown identifier {}", name)
    }

    pub fn push_scope(&mut self) {
        self.list.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.list.len() < 2 {
            unreachable!();
        }
        self.list.pop();
    }
}
