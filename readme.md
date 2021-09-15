# compiler-rs
Compiler for a terrible programming language, written in Rust.

It was originally intended as an IL for a C-like language compiler, but I can
hardly call it that with its high-level expression syntax. There is no block
based control flow, only labels and (un-)conditional jumps.

The grammar of the language is defined in EBNF notation in `doc/grammar.txt`.

Current features:
- Strong typing, no implicit type conversions
- Signed and unsigned integer types of various widths
- Record and union types
- Rust inspired sytax
- Assembly generation for x86\_64 (SysV ABI)

Notably missing features:
- Any "real" optimizations (we do constant folding, and temporaries use a register allocator)
- Struct return types and arguments (struct pointers work fine)
- Floating point math

## license
GPLv2 only.
