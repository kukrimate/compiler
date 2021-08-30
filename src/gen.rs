use super::ast::IntVal;
use super::ast::Type;

//
// Pointer size for the target arch
//

pub const PTR_SIZE: usize = 8;

//
// Storage stroage location
//

#[derive(Clone,Copy,Debug)]
pub enum Section {
    Code,
    Rodata,
    Data,
    Bss
}

#[derive(Clone,Copy,Debug)]
pub struct Storage {
    section: Section,
    offset: usize,
}

//
// Relocation
//

struct Reloc {
    loc: Storage,   // Location of the relocation
    dest: Storage,  // Where the relocation points
}

//
// Code generator interface
//

pub struct Gen {
    // Relocation database
    relocs: Vec<Reloc>,
    // Sections
    rodata: Vec<u8>,
    data: Vec<u8>,
    bss_size: usize,
}

impl Gen {
    pub fn new() -> Gen {
        Gen {
            relocs: Vec::new(),
            rodata: Vec::new(),
            data: Vec::new(),
            bss_size: 0,
        }
    }

    fn put_bytes(&mut self, bytes: &[u8], rw: bool) -> Storage {
        let section = if rw {
            &mut self.data
        } else {
            &mut self.rodata
        };

        let offset = section.len();
        section.extend(bytes);

        if rw {
            Storage { section: Section::Data, offset: offset }
        } else {
            Storage { section: Section::Rodata, offset: offset }
        }
    }

    pub fn put_intval(&mut self, intval: IntVal, rw: bool) -> Storage {
        match intval {
            IntVal::U8(v)  => self.put_bytes(&v.to_le_bytes(), rw),
            IntVal::I8(v)  => self.put_bytes(&v.to_le_bytes(), rw),
            IntVal::U16(v) => self.put_bytes(&v.to_le_bytes(), rw),
            IntVal::I16(v) => self.put_bytes(&v.to_le_bytes(), rw),
            IntVal::U32(v) => self.put_bytes(&v.to_le_bytes(), rw),
            IntVal::I32(v) => self.put_bytes(&v.to_le_bytes(), rw),
            IntVal::U64(v) => self.put_bytes(&v.to_le_bytes(), rw),
            IntVal::I64(v) => self.put_bytes(&v.to_le_bytes(), rw),
        }
    }

    pub fn put_ptr(&mut self, ptr: Storage, rw: bool) -> Storage {
        let pad = [0u8; PTR_SIZE];
        let loc = self.put_bytes(&pad, rw);
        self.relocs.push(Reloc { loc: loc, dest: ptr });
        loc
    }

    pub fn bss_alloc(&mut self, r#type: Type) -> Storage {
        let offset = self.bss_size;
        self.bss_size += r#type.get_size();
        Storage { section: Section::Bss, offset: offset }
    }
}
