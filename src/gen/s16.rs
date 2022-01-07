// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation backend for Sigma16
// Link: https://sigma16.herokuapp.com/build/3.4.0/Sigma16/Sigma16.html
//

use crate::ast::{Ty,Vis};
use super::{DataGen,UOp,BOp,Cond,FuncGen,Gen};

use std::fmt::{Display,Formatter,Write};
use std::rc::Rc;

//
// Registers (R0-R15)
//  - R0  - always zero
//  - R13 - stack pointer
//  - R14 - return address
//  - R15 - arithmetic flags
//

#[derive(Clone,Copy)]
struct Reg(usize);

impl Display for Reg {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "R{}", self.0)
    }
}

//
// Unique labels
//

#[derive(Clone,Copy)]
pub struct Label(usize);

impl Display for Label {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "{}", self.0)
    }
}

//
// Promise for a runtime value
//

#[derive(Clone,Debug,Hash,PartialEq,Eq)]
pub enum Val {
    Void,                     // Non-existent value
    Imm(usize),               // Immediate constant
    Loc(usize),               // Reference to stack
    Glo(Rc<str>, usize),      // Reference to symbol
    Deref(Box<Val>, usize),   // De-reference of pointer
}

impl Val {
    fn with_offset(self, add: usize) -> Val {
        match self {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Loc(offset) => Val::Loc(offset + add),
            Val::Glo(name, offset) => Val::Glo(name, offset + add),
            Val::Deref(ptr, offset) => Val::Deref(ptr, offset + add),
        }
    }
}

//
// Code generator implementation
//

pub struct DataGenS16 {
    data: String
}

impl DataGen for DataGenS16 {
    fn _const(&mut self, _ty: &Ty, val: usize) {
        writeln!(self.data, " data {}", val).unwrap();
    }

    fn addr(&mut self, name: &Rc<str>) {
        writeln!(self.data, " data {}", name).unwrap();
    }
}

pub struct FuncGenS16 {
    frame_size: usize,
    arg_idx: usize,
    label_no: usize,
    code: String,
}

impl FuncGenS16 {
    fn stack_alloc(&mut self, size: usize) -> usize {
        // FIXME: align allocation
        let offset = self.frame_size;
        self.frame_size += size;
        offset
    }

    fn next_ptr(&mut self) -> Val {
        Val::Loc(self.stack_alloc(1))
    }

    // Load the address of a value into a register
    fn gen_lea(&mut self, dreg: Reg, val: &Val) {
        match val {
            Val::Void | Val::Imm(_)
                => unreachable!(),
            Val::Loc(offset)
                => writeln!(self.code, " lea {},{}[R13]", dreg, offset).unwrap(),
            Val::Glo(name, offset) => {
                assert!(*offset == 0); // FIXME: john's assembler errors on expressions
                writeln!(self.code, " lea {},{}[R0]", dreg, name).unwrap()
            },
            Val::Deref(ptr, offset) => {
                self.gen_load(dreg, ptr);
                if *offset > 0 {
                    writeln!(self.code, " lea {},{}[{}]", dreg, offset, dreg).unwrap()
                }
            },
        }
    }

    // Load a value with a certain width to a register
    fn gen_load(&mut self, dreg: Reg, val: &Val) {
        match val {
            Val::Void
                => unreachable!(),
            Val::Imm(val)
                => writeln!(self.code, " lea {},{}[R0]", dreg, val).unwrap(),
            Val::Loc(offset)
                => writeln!(self.code, " load {},{}[R13]", dreg, offset).unwrap(),
            Val::Glo(name, offset) => {
                assert!(*offset == 0);
                writeln!(self.code, " load {},{}[R0]", dreg, name).unwrap()
            },
            Val::Deref(ptr, offset) => {
                self.gen_load(dreg, ptr);
                writeln!(self.code, " load {},{}[{}]", dreg, offset, dreg).unwrap()
            },
        }
    }

    // Store a register's contents into memory
    fn gen_store(&mut self, val: &Val, sreg: Reg, tmp_reg: Reg) {
        match val {
            Val::Void | Val::Imm(_)
                => unreachable!(),
            Val::Loc(offset)
                => writeln!(self.code, " store {},{}[R13]", sreg, offset).unwrap(),
            Val::Glo(name, offset) => {
                assert!(*offset == 0);
                writeln!(self.code, " store {},{}[R0]", sreg, name).unwrap()
            },
            Val::Deref(ptr, offset) => {
                self.gen_load(tmp_reg, ptr);
                writeln!(self.code, " store {},{}[{}]", sreg, offset, tmp_reg).unwrap();
            },
        }
    }

    // Copy size bytes between two locations
    fn gen_copy(&mut self, mut dst: Val, mut src: Val, mut size: usize) {
        while size > 0 {
            // Do the copy
            self.gen_load(Reg(1), &src);
            self.gen_store(&dst, Reg(1), Reg(2));

            // Adjust for the next step
            size -= 1;
            if size > 0 {
                src = src.with_offset(1);
                dst = dst.with_offset(1);
            }
        }
    }
}

impl FuncGen for FuncGenS16 {
    type Label = Label;
    type Val = Val;

    fn next_local(&mut self, ty: &Ty) -> Val {
        Val::Loc(self.stack_alloc(ty.get_size()))
    }

    fn next_label(&mut self) -> Label {
        let no = self.label_no;
        self.label_no += 1;
        Label(no)
    }

    fn tgt(&mut self, label: Label) {
        writeln!(self.code, "L{}", label).unwrap();
    }

    fn jmp(&mut self, label: Label) {
        writeln!(self.code, " jump L{}[R0]", label).unwrap();
    }

    fn arg_to_val(&mut self, ty: &Ty, dst: Val) {
        // FIXME: the calling convention should fall back to the stack
        // FIXME2: larger than register arguments should be handled
        assert!(self.arg_idx < 8);
        assert!(ty.get_size() == 1);
        // NOTE: the temporary register is never used by these stores
        self.gen_store(&dst, Reg(self.arg_idx + 1), Reg(0));
        self.arg_idx += 1;
    }

    fn val_to_ret(&mut self, ty: &Ty, src: Val) {
        assert!(ty.get_size() == 1);
        // FIXME: larger than word return values should be supported
        self.gen_load(Reg(1), &src);
    }

    fn bool_to_val(&mut self, ltrue: Label, lfalse: Label) -> Val {
        let val = self.next_local(&Ty::Bool);
        let lend = self.next_label();
        // True case
        writeln!(self.code, "L{}", ltrue).unwrap();
        self.gen_load(Reg(1), &Val::Imm(1));
        self.gen_store(&val, Reg(1), Reg(2));
        writeln!(self.code, " jump L{}[R0]", lend).unwrap();
        // False case
        writeln!(self.code, "L{}", lfalse).unwrap();
        self.gen_store(&val, Reg(0), Reg(2));
        writeln!(self.code, "L{}", lend).unwrap();
        val
    }

    fn val_to_bool(&mut self, val: Val, ltrue: Label, lfalse: Label) {
        self.gen_load(Reg(1), &val);
        writeln!(self.code, " sub R1,R1,R1").unwrap();
        writeln!(self.code, " jumpz L{}[R0]\n jump L{}[R0]", ltrue, lfalse).unwrap();
    }

    fn immed(&mut self, val: usize) -> Val {
        Val::Imm(val)
    }

    fn global(&mut self, name: Rc<str>) -> Val {
        Val::Glo(name, 0)
    }

    fn compound(&mut self, ty: &Ty, vals: Vec<(&Ty, Val)>) -> Val {
        let off = self.stack_alloc(ty.get_size());
        let mut cur = off;
        for (ty, val) in vals.into_iter() {
            self.gen_copy(Val::Loc(cur), val, ty.get_size());
            cur += ty.get_size();
        }
        Val::Loc(off)
    }

    fn field(&mut self, val: Val, off: usize) -> Val {
        val.with_offset(off)
    }

    fn elem(&mut self, array: Val, index: Val, stride: usize) -> Val {
        if let Val::Imm(index) = index {
            // Avoid emitting multiply on constant index
            array.with_offset(index * stride)
        } else {
            // Generate a de-reference lvalue from a pointer to the element
            self.gen_lea(Reg(1), &array);
            self.gen_load(Reg(2), &index);
            self.gen_load(Reg(3), &Val::Imm(stride));
            // Multiply index by stride
            writeln!(self.code, " mul R2,R2,R3").unwrap();
            writeln!(self.code, " add R1,R1,R2").unwrap();

            // Allocate temporary
            let tmp = self.next_ptr();
            self.gen_store(&tmp, Reg(1), Reg(2));
            Val::Deref(Box::new(tmp), 0)
        }
    }

    fn call(&mut self, retty: &Ty, _varargs: bool, func: Val, args: Vec<(&Ty, Val)>) -> Val {
        // FIXME: non-register arguments
        assert!(args.len() < 8);

        // Move arguments to registers
        for (i, (ty, val)) in args.iter().enumerate() {
            // FIXME: larger arguments
            assert!(ty.get_size() == 1);
            self.gen_load(Reg(i+1), val);
        }

        match func {
            // Special case for interrupts :|)
            Val::Glo(val, 0) if &*val == "trap"
                => writeln!(self.code, " trap R1,R2,R3").unwrap(),
            _ => {
                // Load function address to R10
                self.gen_lea(Reg(10), &func);
                // Generate call
                writeln!(self.code, " jal R14,0[R10]").unwrap();
            }
        }


        if let Ty::Void = retty {
            // Create unusable value for precude
            Val::Void
        } else {
            // Otherwise move returned value to temporary
            let tmp = self.next_local(retty);
            self.gen_store(&tmp, Reg(1), Reg(2));
            tmp
        }
    }

    fn _ref(&mut self, val: Val) -> Val {
        // Load address to rax
        self.gen_lea(Reg(1), &val);
        // Save pointer to temporary
        let tmp = self.next_ptr();
        self.gen_store(&tmp, Reg(1), Reg(2));
        tmp
    }

    fn deref(&mut self, val: Val) -> Val {
        Val::Deref(Box::new(val), 0)
    }

    fn unary(&mut self, _op: UOp, _ty: &Ty, _val: Val) -> Val {
        todo!()
    }

    fn binary(&mut self, op: BOp, ty: &Ty, v1: Val, v2: Val) -> Val {
        // Load operands into registers
        self.gen_load(Reg(1), &v1);
        self.gen_load(Reg(2), &v2);

        match op {
            BOp::Mul => {
                // We only care about the lower half of the result
                // Sigma16 has no unsigned multiply anyways, but the
                // result is always identical
                writeln!(self.code, " mul R1,R1,R2").unwrap()
            },
            BOp::Div | BOp::Rem => {
                if !ty.is_signed() {
                    // Oh John, why are you doing this to us, I just want to
                    // divide two gosh darn unsigned words :<(
                    panic!("Sigma16 cannot do unsigned divide")
                }

                writeln!(self.code, " div R1,R1,R2").unwrap()
            },

            // These operations are sign independent with two's completement
            BOp::Add => writeln!(self.code, " add R1,R1,R2").unwrap(),
            BOp::Sub => writeln!(self.code, " sub R1,R1,R2").unwrap(),

            // The right operand can be any integer type, however it must have
            // a positive value less-than the number of bits in the left operand
            // otherwise this operation is undefined behavior
            BOp::Lsh => writeln!(self.code, " lsh R1,R1,R2").unwrap(),
            BOp::Rsh => writeln!(self.code, " rsh R1,R1,R2").unwrap(),

            // These operations are purely bitwise and ignore signedness
            BOp::And => writeln!(self.code, " and R1,R1,R2").unwrap(),
            BOp::Xor => writeln!(self.code, " xor R1,R1,R2").unwrap(),
            BOp::Or  => writeln!(self.code, " or R1,R1,R2").unwrap(),
        }

        // Save result to temporary
        let tmp = self.next_local(ty);
        if op == BOp::Rem {
            // Remainder always in Reg(15)
            self.gen_store(&tmp, Reg(15), Reg(3));
        } else {
            self.gen_store(&tmp, Reg(1), Reg(3));
        }
        tmp
    }

    fn jcc(&mut self, cond: Cond, ltrue: Label, lfalse: Label, ty: &Ty, v1: Val, v2: Val) {
        self.gen_load(Reg(1), &v1);
        self.gen_load(Reg(2), &v2);

        // Generate compare
        writeln!(self.code, " cmp R1,R2").unwrap();
        // Generate conditional jump to true case
        match cond {
            Cond::Lt => if ty.is_signed() {
                writeln!(self.code, " jumplt L{}[R0]", ltrue).unwrap();
            } else {
                panic!("Sigma16 cannot do unsigned relations")
            },
            Cond::Le => if ty.is_signed() {
                writeln!(self.code, " jumple L{}[R0]", ltrue).unwrap();
            } else {
                panic!("Sigma16 cannot do unsigned relations")
            },
            Cond::Gt => if ty.is_signed() {
                writeln!(self.code, " jumpgt L{}[R0]", ltrue).unwrap();
            } else {
                panic!("Sigma16 cannot do unsigned relations")
            },
            Cond::Ge => if ty.is_signed() {
                writeln!(self.code, " jumpge L{}[R0]", ltrue).unwrap();
            } else {
                panic!("Sigma16 cannot do unsigned relations")
            },
            Cond::Eq => writeln!(self.code, " jumpeq L{}[R0]", ltrue).unwrap(),
            Cond::Ne => writeln!(self.code, " jumpne L{}[R0]", ltrue).unwrap(),
        }
        // Generate unconditional jump to false case
        writeln!(self.code, " jump L{}[R0]", lfalse).unwrap();
    }

    fn assign(&mut self, ty: &Ty, dst: Val, src: Val) {
        self.gen_copy(dst, src, ty.get_size());
    }
}

pub struct GenS16 {
    // Global label no, s16asm has no local labels :/
    label_no: usize,
    // Linkage table
    str_no: usize,
    // Sections
    code: String,
    data: String,
}

impl GenS16 {
    pub fn new() -> GenS16 {
        GenS16 {
            label_no: 0,
            str_no: 0,
            code: String::new(),
            data: String::new(),
        }
    }

    pub fn finalize<T: std::io::Write>(&self, output: &mut T) {
        // Setup code goes at address zero before everything

        writeln!(output, " add R13,R0,R0").unwrap();    // Stack grows down from the top of memory
        writeln!(output, " jal R14,main[R0]").unwrap(); // Then we call main
        writeln!(output, " trap R0,R0,R0").unwrap();    // After main returns we exit

        // Then we first write our functions
        writeln!(output, "{}", self.code).unwrap();
        // And finally our data
        writeln!(output, "{}", self.data).unwrap();
    }
}

impl Gen for GenS16 {
    type DataGen = DataGenS16;
    type FuncGen = FuncGenS16;

    fn do_link(&mut self, _name: Rc<str>, _vis: Vis) {
        // no-op, we only care about simple programs for Sigma16
    }

    fn do_string(&mut self, chty: Ty, data: &str) -> (Rc<str>, Ty) {
        // Create assembly symbol
        let name: Rc<str> = format!("str{}", self.str_no).into();
        self.str_no += 1;
        // Generate data
        writeln!(self.data, "{}", name).unwrap();
        for byte in data.bytes() {
            writeln!(self.data, " data {}", byte).unwrap();
        }
        writeln!(self.data, " data 0").unwrap();
        // Insert symbol
        let ty = Ty::Array {
            elem_type: Box::new(chty),
            elem_count: Some(data.len())
        };
        (name, ty)
    }

    fn do_bss(&mut self, name: &Rc<str>, ty: &Ty) {
        // Allocate bss entry

        // The Sigma16 assembler doesn't have the concept of reserving storage,
        // we just fill with zeroes
        writeln!(self.data, "{}", name).unwrap();
        for _ in 0..ty.get_size() {
            writeln!(self.data, " data 0").unwrap();
        }

    }

    fn begin_data(&mut self, name: &Rc<str>) -> DataGenS16 {
        // Generate heading
        writeln!(self.data, "{}", name).unwrap();
        // Make data generator
        DataGenS16 {
            data: String::new()
        }
    }

    fn end_data(&mut self, data_gen: DataGenS16) {
        // Merge generated data
        write!(self.data, "{}", data_gen.data).unwrap();
    }

    fn begin_func(&mut self, name: &Rc<str>) -> FuncGenS16 {
        // Create function heading
        writeln!(self.code, "{}", name).unwrap();
        // Create generation context
        FuncGenS16 {
            frame_size: 0,
            arg_idx: 0,
            label_no: self.label_no,
            code: String::new(),
        }
    }

    fn end_func(&mut self, func_gen: FuncGenS16) {
        self.label_no = func_gen.label_no;
        // Entry sequence
                                                            // Allocate space
        writeln!(self.code, " lea R11,{}[R0]", func_gen.frame_size + 1).unwrap();
        writeln!(self.code, " sub R13,R13,R11").unwrap();
        writeln!(self.code, " store R14,{}[R13]", func_gen.frame_size).unwrap();

        // Function body
        write!(self.code, "{}", func_gen.code).unwrap();

        // Exit sequence
        writeln!(self.code, " load R14,{}[R13]", func_gen.frame_size).unwrap();
                                                            // Release space
        writeln!(self.code, " lea R11,{}[R0]", func_gen.frame_size + 1).unwrap();
        writeln!(self.code, " add R13,R13,R11").unwrap();
        writeln!(self.code, " jump 0[R14]").unwrap();       // return
    }
}
